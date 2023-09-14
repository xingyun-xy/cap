"""predict tools."""
import onnxruntime
import tensorrt as trt
# print(trt.__version__)
import numpy as np

import sys
sys.path.append('/workspace/cap_develop/')

import argparse
import os
import copy
import warnings
from typing import Sequence, Union
import sys
sys.path.append("/code/cap_develop_zhangwenjie/")

import onnx_graphsurgeon as gs

import changan_plugin_pytorch as changan
from changan_plugin_pytorch.utils.onnx_helper import export_to_onnx
import torch
import torch.onnx
import torch.nn as nn
from onnxsim import simplify
import onnx

from cap.engine import build_launcher
from cap.registry import RegistryContext, build_from_registry
from cap.utils.config import ConfigVersion
from cap.utils.config_v2 import Config
from cap.utils.distributed import get_dist_info
from cap.utils.logger import (
    DisableLogger,
    MSGColor,
    format_msg,
    rank_zero_info,
)
from capbc.utils import deprecated_warning
from utilities import LOG_DIR, init_rank_logger
from cap.models.model_convert.converters import LoadCheckpoint
from cap.models.model_convert.pipelines import ModelConvertPipeline


debug_engine = False


def build_engine(onnx_file_path, enable_fp16=False, max_batch_size=1, max_workspace_size=10, write_engine=True):
    graph = gs.import_onnx(onnx.load(onnx_file_path))
    precision_name_list = list()

    onnx_path = os.path.realpath(onnx_file_path) 
    engine_file_path = ".".join(onnx_path.split('.')[:-1] + ['engine' if not enable_fp16 else 'fp16.engine'])
    print('engine_file_path', engine_file_path)
    G_LOGGER = trt.Logger(trt.Logger.WARNING)
    if os.path.exists(engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine, engine_file_path
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, G_LOGGER) as parser:

        config = builder.create_builder_config()
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(max_workspace_size))
        if enable_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        print('Loading ONNX file from path {} ...'.format(onnx_file_path))

        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None, None
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)

        if enable_fp16 and builder.platform_has_fast_fp16:
            for i in range(network.num_layers):

                layer = network.get_layer(i)

                layer_type = layer.type

                if layer_type in (trt.LayerType.SHAPE, trt.LayerType.SLICE,
                                  trt.LayerType.IDENTITY,
                                  trt.LayerType.SHUFFLE, trt.LayerType.RESIZE):
                    print(f'{layer.name} passed 1')
                    continue

                layer_output_precision = layer.get_output(0).dtype
                print(f'layer_name: {layer.name}, layer_output_precision: {layer_output_precision}')

                if layer_output_precision in (trt.int32, trt.int8, trt.bool):
                    print(f'{layer.name} passed 2')
                    continue

                if i > 127 and i < 148:
                    print(f'layer {layer.name} set fp32 precision mode')
                    # layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
                    layer.precision = trt.float32

        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            return None, None
        print("Completed creating Engine")
        # 保存engine文件
        if write_engine:
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
        with trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine, engine_file_path


class TRTModule(torch.nn.Module):
    def __init__(self, 
                    engin_path,
                    input_names=["sweep_imgs", "mlp_input", "circle_map", "ray_map"], #["sweep_imgs", "sensor2ego_mats", "intrin_mats", "ida_mats", "bda_mat"], 
                    # input_names=["sweep_imgs"], 
                    output_names=["preds_%d"%i for i in range(1,37)]):
        super(TRTModule, self).__init__()
        
        logger = trt.Logger(trt.Logger.INFO)
        
        with open(engin_path, "rb") as f, trt.Runtime(logger) as runtime: # 转换出来的其实是个引擎，这里使用runtime去序列化这个trt引擎 以读的方式打开引擎，
            self.engine =runtime.deserialize_cuda_engine(f.read())        # 然后构建一个运行时的对象，然后通过这个运行时对象序列化读过来的引擎
            
        if self.engine is not None: # 如果engine确实存在，则创建上下文环境
            self.context = self.engine.create_execution_context() # engine创建执行context 利用引擎创造执行环境的上下文  这里的引擎可以理解成模型的所有的信息 上下文环境理解成forward
        self.input_names = input_names
        self.output_names = output_names

        model_all_names = []
        for idx in range(self.engine.num_bindings): # 遍历这个引擎的绑定 包括输入和输出
            is_input = self.engine.binding_is_input(idx)  # 这个绑定是输入吗？
            name     = self.engine.get_binding_name(idx)  # 这个绑定的名字
            op_type  = self.engine.get_binding_dtype(idx) # 这个绑定的数据类型
            shape    = self.engine.get_binding_shape(idx) # 这个绑定的形状
            model_all_names.append(name)
            print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)
    
    def trt_version(self,): # 返回trt的版本
        return trt.__version__

    def torch_dtype_from_trt(self,dtype): # 根据trt的类型 返回torch对应的类型
        if dtype == trt.int8:
            return torch.int8
        elif self.trt_version() >= '7.0' and dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError("%s is not supported by torch" % dtype)

    def torch_device_from_trt(self,device): # 根据trt对应的设备，返回torch对应的设备
        if device == trt.TensorLocation.DEVICE:
            return torch.device("cuda")
        elif device == trt.TensorLocation.HOST:
            return torch.device("cpu")
        else:
            return TypeError("%s is not supported by torch" % device)
        
    def forward(self, sweep_imgs, mlp_input=None, circle_map=None, ray_map=None):
        
        if mlp_input is None and circle_map is None and ray_map is None:
            inputs = [sweep_imgs]
        else:
            inputs = [sweep_imgs, mlp_input, circle_map, ray_map]
        batch_size = inputs[0].shape[0] # 拿到输入的batch
        bindings = [None] * (len(self.input_names) + len(self.output_names)) # 4个输入 36个输出，那就是 40 个 None 元素

        for i, input_name in enumerate(self.input_names): # 遍历所有的输入名字
            idx = self.engine.get_binding_index(input_name) # 拿到这个输入名字的索引
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))# 设定shape 这应该意味着传进来的输入顺序就是对应trt中的输入顺序
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        
        for i, output_name in enumerate(self.output_names): # 绑定所有的输出
            idx           = self.engine.get_binding_index(output_name)
            dtype         = self.torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape         = tuple(self.context.get_binding_shape(idx))
            device        = self.torch_device_from_trt(self.engine.get_location(idx))
            # 以上三个都作为下边这个API的输入，包括拿到输出的类型、形状、所在设备
            #                  || 
            #                 \||/
            #                  \/
            output        = torch.empty(size=shape, dtype=dtype, device=device) 
            outputs[i]    = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the predict stage, you should define "
            "{stage}_predictor in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
        help="GPU device ids like '0,1,2,3, "
        "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="NCCL",
        choices=["NCCL", "GLOO"],
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    
    return parser.parse_args()


# 状态字典更新函数
def state_dict_update(state_dict):
    new_state_dict = {k: torch.tensor(v,dtype=torch.float32) for k, v in state_dict.items()}
    for k,v in new_state_dict.items():
        assert isinstance(v,torch.FloatTensor),"不是浮点tensor"
    return new_state_dict

def pytorch2onnx(
    device: Union[None, int, Sequence[int]],
    stage: str,
    cfg_file: str,
):
    cfg = Config.fromfile(cfg_file)
    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step="onnx",
        prefix="pytorch2onnx-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    rank_zero_info("=" * 50 + "BEGIN PYTORCH2ONNX" + "=" * 50)

    # 2. build model
    print (cfg.onnx_cfg['checkpoint_path'])
    model_cfg = copy.deepcopy(cfg.onnx_cfg['export_model'])
    model = build_from_registry(model_cfg)
    
    convert_pipeline = ModelConvertPipeline(
        converters=[
            LoadCheckpoint(cfg.onnx_cfg['checkpoint_path'],
            # state_dict_update_func=state_dict_update,  #状态字典更新函数
            allow_miss=True,
            ignore_extra=True),
        ],
    )
    model = convert_pipeline(model)
    model.eval()

    # 3. pytorch2onnx
    onnx_name = cfg.onnx_cfg['onnx_name'] + '.onnx'  #onnx保存路径
    input_names = ["input","sensor2ego_mats","intrin_mats","ida_mats","sensor2sensor_mats","bda_mat","mlp_input","circle_map","ray_map"]
    """
    img=torch.rand((6, 3, 320, 576)), #修改输入尺寸
    # img=torch.rand((6, 3, 512, 960)), #修改输入尺寸
    sensor2ego_mats=torch.rand((1, 1, 6, 4, 4)),
    intrin_mats=torch.rand((1, 1, 6, 4, 4)),
    ida_mats=torch.rand((1, 1, 6, 4, 4)),
    sensor2sensor_mats=torch.rand((1, 1, 6, 4, 4)),
    bda_mat=torch.rand((1, 4, 4)),
    mlp_input = torch.rand((1,1,6,27)),
    circle_map = torch.rand((1,112,16384)),
    ray_map=torch.rand((1,216,16384))
    
    
    """
    # dummy_input = cfg.onnx_cfg['dummy_input']['img']
    dummy_input = cfg.onnx_cfg['dummy_input']
    data = dummy_input
    output_names = cfg.onnx_cfg['output_names']

    output_names = [
                    "bev_reg00",
                    "bev_height00",
                    "bev_dim00",
                    "bev_rot00",
                    "bev_vel00",
                    "bev_heatmap00",

                    "bev_reg01",
                    "bev_height01",
                    "bev_dim01",
                    "bev_rot01",
                    "bev_vel01",
                    "bev_heatmap01",

                    "bev_reg02",
                    "bev_height02",
                    "bev_dim02",
                    "bev_rot02",
                    "bev_vel02",
                    "bev_heatmap02",

                    "bev_reg03",
                    "bev_height03",
                    "bev_dim03",
                    "bev_rot03",
                    "bev_vel03",
                    "bev_heatmap03",

                    "bev_reg04",
                    "bev_height04",
                    "bev_dim04",
                    "bev_rot04",
                    "bev_vel04",
                    "bev_heatmap04",
                    
                    "bev_reg05",
                    "bev_height05",
                    "bev_dim05",
                    "bev_rot05",
                    "bev_vel05",
                    "bev_heatmap05",]

    # input_names_dict = {'input':{0:'batch_size', 2:'height', 3:'width'}}
    input_names_dict = {'input':{0:'batch_size'}}
    output_names_dict = {'vehicle_rpn_pred': {0:'batch_size'},
                         'vehicle_detection_roi_split_head.box_score': {0:'batch_size'},
                         'vehicle_detection_roi_split_head.box_reg': {0:'batch_size'},
                         'vehicle_category_classification_roi_split_head.box_score': {0:'batch_size'},
                         'x': {0:'batch_size'},
                         'vehicle_occlusion_classification_roi_split_head.box_score': {0:'batch_size'},
                         'vehicle_wheel_kps_roi_split_head.label_out_block': {0:'batch_size'},
                         'vehicle_wheel_kps_roi_split_head.pos_offset_out_block': {0:'batch_size'},
                         'rear_rpn_pred': {0:'batch_size'},
                         'rear_detection_roi_split_head.box_score': {0:'batch_size'},
                         'rear_detection_roi_split_head.box_reg': {0:'batch_size'},
                         'rear_occlusion_classification_roi_split_head.box_score': {0:'batch_size'},
                         'rear_part_classification_roi_split_head.box_score': {0:'batch_size'},
                         'lane_parsing': {0:'batch_size'},
                         'parsing': {0:'batch_size'},}

    # input_names_dict = {'input':{0:6}}
    # output_names_dict = {'vehicle_rpn_pred': {0:6},
    #                      'vehicle_detection_roi_split_head.box_score': {0:6},
    #                      'vehicle_detection_roi_split_head.box_reg': {0:6},
    #                      'vehicle_category_classification_roi_split_head.box_score': {0:6},
    #                      'x': {0:6},
    #                      'vehicle_occlusion_classification_roi_split_head.box_score': {0:6},
    #                      'vehicle_wheel_kps_roi_split_head.label_out_block': {0:6},
    #                      'vehicle_wheel_kps_roi_split_head.pos_offset_out_block': {0:6},
    #                      'rear_rpn_pred': {0:6},
    #                      'rear_detection_roi_split_head.box_score': {0:6},
    #                      'rear_detection_roi_split_head.box_reg': {0:6},
    #                      'rear_occlusion_classification_roi_split_head.box_score': {0:6},
    #                      'rear_part_classification_roi_split_head.box_score': {0:6},
    #                      'lane_parsing': {0:6},
    #                      'parsing': {0:6},}
    # dynamic_axes = {**input_names_dict, **output_names_dict}

    if debug_engine:
        
        build_engine("panorama_multitask_resize.onnx", enable_fp16=True)

        sweep_imgs = np.loadtxt("bev_input/input_4.txt")
        sweep_imgs = sweep_imgs.reshape((6,3,320,576)).astype(np.float32)
        sweep_imgs = torch.from_numpy(sweep_imgs).cuda()
        mlp_input = np.loadtxt("bev_input/mlp_input.txt")
        mlp_input = mlp_input.reshape((1,1,6,27)).astype(np.float32)
        mlp_input = torch.from_numpy(mlp_input).cuda()
        circle_map = np.loadtxt("bev_input/circle_map.txt")
        circle_map = circle_map.reshape((1,112,16384)).astype(np.float32)
        circle_map = torch.from_numpy(circle_map).cuda()
        ray_map = np.loadtxt("bev_input/ray_map.txt")
        ray_map = ray_map.reshape((1,216,16384)).astype(np.float32)
        ray_map = torch.from_numpy(ray_map).cuda()

        trt_model = TRTModule(engin_path="/code/cap_develop_zhangwenjie/panorama_multitask_resize.fp16.engine", 
                            input_names=["input", "mlp_input", "circle_map", "ray_map"], output_names=output_names)

        result_trt = trt_model(sweep_imgs, mlp_input, circle_map, ray_map)
    
        preds = [[{'reg':result_trt[0], 'height':result_trt[1], 'dim':result_trt[2], 'rot':result_trt[3], 'vel':result_trt[4], 'heatmap':result_trt[5]}],
                [{'reg':result_trt[6], 'height':result_trt[7], 'dim':result_trt[8], 'rot':result_trt[9], 'vel':result_trt[10],'heatmap':result_trt[11]}],
                [{'reg':result_trt[12],'height':result_trt[13],'dim':result_trt[14],'rot':result_trt[15],'vel':result_trt[16],'heatmap':result_trt[17]}],
                [{'reg':result_trt[18],'height':result_trt[19],'dim':result_trt[20],'rot':result_trt[21],'vel':result_trt[22],'heatmap':result_trt[23]}],
                [{'reg':result_trt[24],'height':result_trt[25],'dim':result_trt[26],'rot':result_trt[27],'vel':result_trt[28],'heatmap':result_trt[29]}],
                [{'reg':result_trt[30],'height':result_trt[31],'dim':result_trt[32],'rot':result_trt[33],'vel':result_trt[34],'heatmap':result_trt[35]}]]

        np.save("bev_output/preds.npy", preds)

        # provider = onnxruntime.get_available_providers()[1 if onnxruntime.get_device() == "GPU" else 0]
        # session = onnxruntime.InferenceSession("panorama_multitask_resize.onnx", providers=[provider])
        # input_names = session.get_inputs()
        # output_names = session.get_outputs()
        # output_names_list = [output_name.name for output_name in output_names]

        # onnx_preds = session.run(output_names = output_names_list, 
        #                         input_feed = {input_names[0].name:sweep_imgs.cpu().numpy(), 
        #                         input_names[1].name:mlp_input.cpu().numpy(),
        #                         input_names[2].name:circle_map.cpu().numpy(),
        #                         input_names[3].name:ray_map.cpu().numpy(),})
        # preds = [[{'reg':torch.from_numpy(onnx_preds[0]), 'height':torch.from_numpy(onnx_preds[1]), 'dim':torch.from_numpy(onnx_preds[2]), 'rot':torch.from_numpy(onnx_preds[3]), 'vel':torch.from_numpy(onnx_preds[4]), 'heatmap':torch.from_numpy(onnx_preds[5])}],
        #         [{'reg':torch.from_numpy(onnx_preds[6]), 'height':torch.from_numpy(onnx_preds[7]), 'dim':torch.from_numpy(onnx_preds[8]), 'rot':torch.from_numpy(onnx_preds[9]), 'vel':torch.from_numpy(onnx_preds[10]),'heatmap':torch.from_numpy(onnx_preds[11])}],
        #         [{'reg':torch.from_numpy(onnx_preds[12]),'height':torch.from_numpy(onnx_preds[13]),'dim':torch.from_numpy(onnx_preds[14]),'rot':torch.from_numpy(onnx_preds[15]),'vel':torch.from_numpy(onnx_preds[16]),'heatmap':torch.from_numpy(onnx_preds[17])}],
        #         [{'reg':torch.from_numpy(onnx_preds[18]),'height':torch.from_numpy(onnx_preds[19]),'dim':torch.from_numpy(onnx_preds[20]),'rot':torch.from_numpy(onnx_preds[21]),'vel':torch.from_numpy(onnx_preds[22]),'heatmap':torch.from_numpy(onnx_preds[23])}],
        #         [{'reg':torch.from_numpy(onnx_preds[24]),'height':torch.from_numpy(onnx_preds[25]),'dim':torch.from_numpy(onnx_preds[26]),'rot':torch.from_numpy(onnx_preds[27]),'vel':torch.from_numpy(onnx_preds[28]),'heatmap':torch.from_numpy(onnx_preds[29])}],
        #         [{'reg':torch.from_numpy(onnx_preds[30]),'height':torch.from_numpy(onnx_preds[31]),'dim':torch.from_numpy(onnx_preds[32]),'rot':torch.from_numpy(onnx_preds[33]),'vel':torch.from_numpy(onnx_preds[34]),'heatmap':torch.from_numpy(onnx_preds[35])}]]
        # np.save("bev_output/preds.npy", preds)

    export_to_onnx(
        model,
        data,
        onnx_name,
        input_names=input_names,
        export_params=True,
        do_constant_folding=True,
        output_names=output_names,
        opset_version=12,
        verbose="True",
        # dynamic_axes=dynamic_axes
    )

    # 4. simplify onnx
    rank_zero_info("=" * 50 + "ONNX_SIMPLIFY" + "=" * 50)

    onnx_model = onnx.load(onnx_name)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_name)
    # add dim
    #onnx.save(onnx.shape_inference.infer_shapes(model_simp),onnx_name)

    rank_zero_info("=" * 50 + "END PYTORCH2ONNX" + "=" * 50)


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    # check config version
    config_version = config.get("VERSION", None)
    if config_version is not None:
        assert (
            config_version == ConfigVersion.v2
        ), "{} only support config with version 2, not version {}".format(
            os.path.basename(__file__), config_version.value
        )
    else:
        warnings.warn(
            "VERSION will must set in config in the future."
            "You can refer to configs/classification/resnet18.py,"
            "and configs/classification/bernoulli/mobilenetv1.py."
        )
    if args.device_ids is not None:
        ids = list(map(int, args.device_ids.split(",")))
    else:
        ids = config.device_ids
    num_processes = config.get("num_processes", None)

    predictor_cfg = config[f"{args.stage}_predictor"]
    launch = build_launcher(predictor_cfg)
    launch(
        pytorch2onnx,
        ids,
        dist_url=args.dist_url,
        dist_launcher=args.launcher,
        num_processes=num_processes,
        backend=args.backend,
        args=(args.stage, args.config),
    )
