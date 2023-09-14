# 产看文件夹内文件个数
# ls -lR|grep "^-"|wc -l

# cam1  左前
# cam2  左后
# cam3  右前
# cam4  右后
# cam5  正后

# 0-result 右前 对应cam3
# 1-result 右后 对应cam4
# 2-result 左前 对应cam1
# 3-result 左后 对应cam2
# 4-result 正后 对应cam5


import os
import json
import shutil
import argparse
from glob import glob  # 获取全局图片
from tqdm import tqdm

CLASS_MAP = {
    '轿车':'Sedan_Car',
    '越野车':'SUV',
    '公交车-大客车':'Bus',
    '大货车':'BigTruck',
    '小货车':'Lorry',
    '自行车':'Bike',
    '面包车':'MiniVan',
    '专用作业车':'Special_vehicle',
    '摩托车、电动车':'Motorcycle',
    '人力三轮车':'Tricycle',
    '电动三轮车、摩托三轮车':'Motor-Tricycle',
    '机动车其它':'Vehicle_others',
    '非机动车其它':'Non-Vehicle_others',
    '微型车':'Tiny_car',
    '不确定':'unknown',
    '车灯类':'Vehicle_light'
}
Corner_confidence={
        "高": "high",
        "中": "middle",
        "低": "low",
        "非常低": "super_low"
    }
truncation_map = {
    '非截断':'None',
    '有效部分高':'High',
    '有效部分中':'Middle',
    '有效部分低':'Low',
    '有效部分非常低':'VeryLow'
}
OCCLUSION_MAP = {
    '完全可见':'full_visible',
    '部分遮挡':'occluded',
    '严重遮挡':'heavily_occluded',
    '完全不可见':'invisible',
    '遮挡':'occluded',
    "图外":"outside" #based on the document of labeling guide, changan dataset contains this, also automatrix labels contains 'self_occluded' attribute.
    
}

CONFIDENCE_MAP = {
    '高': 'High',
    '中': 'Middle',
    '低': 'Low',
    '非常低':'VeryLow' ,
    
}


IGNORE_MAP = {
    '是': 'yes',
    '否': 'no',
}

ORIENTATION_MAP = {
    '正向': 'facade',
    '斜向': 'oblique',
    '横向': 'Transverse',
    '未知': 'unknown'
}

PART_MAP = {
    '车头': 'head',
    '车尾': 'rear'
}


class RearToHorizon(object):
    def __init__(self, data_dir, save_path, log_path):
        self.data_dir = data_dir
        self.data_id = data_dir.split("/")[-1]
        self.save_path = save_path
        self.log_path = log_path

        self.content_error_json_path = []
        self.question_error_json_path = []
        self.data_error_json_path = []
        self.repeat_image_name_list = []
        self.repeat_json_path = []
        self.contain_chinese_json_path = []
        self.effective_json_path = []

        self.total_json_nums = 0
        self.save_json_flag = True
        self.delete_file(self.save_path)
        self.delete_file(self.log_path)
        
    def do_convert(self):
        sub_dir_list = sorted(glob(self.data_dir + '/*'))
        for sub_dir in sub_dir_list:
            sub_dir_list = os.listdir(sub_dir)
            if "annotation" in sub_dir_list:
                self.log2txt(f"================================start!--{sub_dir}================================")
                json_path_list = sorted(glob(sub_dir + '/annotation/**/*.json', recursive=True))
                self.total_json_nums += len(json_path_list)
                for json_path in tqdm(json_path_list):
                    self.save_json_flag = True
                    image_path, target_path, target_data_path, image_name, version_info = self.get_file_info(json_path)
                    # process repeat
                    if image_name in self.effective_json_path: 
                        self.repeat_image_name_list.append(image_name)
                        self.repeat_json_path.append(json_path)
                        self.save_json_flag = False
                        continue
                    # process json
                    with open(json_path, "r", encoding='utf-8') as f:
                        try: # 捕获json 内容错误
                            label = json.load(f)
                        except Exception as e:
                            self.content_error_json_path.append(json_path)
                            self.save_json_flag = False
                            continue
                        new_label = {}
                        vehicle = []
                        belongs_to =[]
                        kps_front_rear = []
                        vehicle_kps_8 =[]
                        kps_id =300001
                        for i, item in enumerate(label["markData"]['annotations']):
                            anno = {}
                            vehicle8 = {}
                            anno["id"] = i + 1
                            bbox_id = anno["id"]
                            anno["track_id"] = -1
                            if item['data'] == []:
                                print('anno_data:',anno['data'])
                                self.data_error_json_path.append(json_path)
                                self.save_json_flag = False
                                continue
                            anno_data = item['data'][0]
                            if len(item)>3:
                                if "question" in item:
                                    kpsattr = list(item['question'].values())
                                if kpsattr == []:
                                    self.data_error_json_path.append(json_path)
                                    self.save_json_flag = False
                                    break
                                bl = f"vehicle|{bbox_id}:vehicle_kps_8|{kps_id}"
                                belongs_to.append(bl)
                                kps_rear = item['data'][1]
                                kps_front = item['data'][2]
                                if kps_front['relativePos'] == {}:
                                    continue
                                if kps_rear['relativePos'] == {}:
                                    continue
                                kps_frontX,kps_frontY =kps_front['relativePos']["x"],kps_front['relativePos']["y"]
                                kps_rearX,kps_rearY =kps_rear['relativePos']["x"],kps_rear['relativePos']["y"]
                                kps_front_rear=[[kps_rearX,kps_rearY],[kps_frontX,kps_frontY]]
                                vehicle8["attr"]=[]
                                vehicle8["data"]=kps_front_rear
                                vehicle8["id"] = kps_id
                                vehicle8["label_type"]="points"
                                vehicle8["luid"]=-1 #dont know the luid :(
                                vehicle8["num"] = 8
                                vehicle8["point_attrs"] = [{"point_label":{"Corner_confidence": Corner_confidence[kpsattr[2]],
                                                            "ignore":IGNORE_MAP[kpsattr[1]], 
                                                            "occlusion":OCCLUSION_MAP[kpsattr[0]], 
                                                            "position":"other"}},
                                                            {"point_label":{"Corner_confidence":Corner_confidence[kpsattr[-1]], 
                                                            "ignore":IGNORE_MAP[kpsattr[-2]], 
                                                            "occlusion":OCCLUSION_MAP[kpsattr[-3]], 
                                                            "position":"other"}}]
                                vehicle8["struct_type"] ="vehicle_kps_8"
                                vehicle8["track_id"] = -1
                                vehicle_kps_8.append(vehicle8)
                                kps_id+=1 
                            anno["data"] = []  
                            anno["attrs"] = {}         
                            try:  # 捕获data为空的json # 解决"relativePos": [] 这个bug
                                assert item['data'][0]['relativePos']
                                bbox_data = item['data'][0]['relativePos']
                                left = bbox_data['left']
                                top = bbox_data['top']
                                right = bbox_data['right']
                                bottom = bbox_data['bottom']
                                anno["data"] = [left, top, right, bottom]
                            except Exception as e:
                                self.data_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break
                            anno["struct_type"] = anno_data['type']
                            try:  # 捕获question为空的json
                                # question_dict = item["data"][0]["question"]
                                ques_list = list(anno_data['question'].values())
                                anno["attrs"]["type"] = CLASS_MAP[ques_list[-1]]
                                anno["attrs"]["ignore"] = IGNORE_MAP[ques_list[1]]
                                anno["attrs"]["occlusion"] = OCCLUSION_MAP[ques_list[0]]
                                anno["attrs"]["confidence"] = CONFIDENCE_MAP[ques_list[2]]
                                anno["attrs"]["truncation"] = truncation_map[ques_list[-2]]
  
                            except Exception as e:
                                self.question_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break
                            
                            vehicle.append(anno)
                    
                    
                    new_label["belong_to"] = belongs_to
                    new_label["height"] = label["markData"]['height']
                    new_label["image_key"] = image_name 
                    new_label["image_source"] = " "
                    new_label["image_uuid"] = "1" 
                    new_label["video_name"] = "1"
                    new_label["video_index"] = "1"
                    new_label["vehicle"] = vehicle
                    new_label["vehicle_kps_8"] = vehicle_kps_8
                    new_label["width"] = label["markData"]['width']

                    if self.save_json_flag:
                        if self.is_contain_chinese(image_name):  
                            self.contain_chinese_json_path.append(json_path)
                        else:
                            self.effective_json_path.append(json_path)
                            self.save_json(target_path, new_label)  # save json
                            shutil.copy(image_path, target_data_path)  # copy image
                        
        total_error_json_nums = len(self.content_error_json_path)+ \
                                len(self.question_error_json_path) + \
                                len(self.data_error_json_path) + \
                                len(self.repeat_image_name_list) + \
                                len(self.contain_chinese_json_path)

        log = f"=========================数据集:{self.data_id}========================= \
            \n====>存在content问题的json_path:{self.content_error_json_path} \
            \n====>存在question问题的json_path:{self.question_error_json_path} \
            \n====>存在data问题的json_path:{self.data_error_json_path} \
            \n====>image_key重复的json_path:{self.repeat_json_path} \
            \n====>image_key包含中文的json_path:{self.contain_chinese_json_path} \
            \n================================================================== \
            \n====>json文件总数:{self.total_json_nums} \
            \n====>有问题json文件总数:{total_error_json_nums} \
            \n     ====>其中json文件content错误总数:{len(self.content_error_json_path)} \
            \n     ====>其中json文件question错误总数:{len(self.question_error_json_path)} \
            \n     ====>其中json文件data错误总数:{len(self.data_error_json_path)} \
            \n     ====>其中json文件repeat总数:{len(self.repeat_image_name_list)} \
            \n     ====>其中json文件image_key包含中文错误总数:{len(self.contain_chinese_json_path)} \
            \n====>有效的json文件总数:{len(self.effective_json_path)} \
            \n====>json文件合格率:{1-total_error_json_nums/self.total_json_nums} \
            \n===============================done!=============================="
        self.log2txt(log)

    def is_contain_chinese(self,check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def get_file_info(self, json_path):
        version_info = json_path.split("/annotation/")[0].split(os.sep)[-1]
        cam_info = json_path.split(os.sep)[-2]
        target_path = os.path.join(self.save_path, self.data_id + "-" + version_info + "-" + cam_info)
   
        target_data_path = os.path.join(target_path, "data")

        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)
        
        image_path = json_path.replace("annotation", "").replace(".json", ".jpg")
        image_name = image_path.split(os.sep)[-1]
       
        assert os.path.exists(image_path), "image_path不存在!"
        return image_path, target_path, target_data_path, image_name, version_info

    def save_json(self, target_path, new_label):
        json_path = target_path + '/data.json'
        with open(json_path, 'a+') as f:
            json_str = json.dumps(new_label)
            f.write(json_str + '\n')

    def log2txt(self,log):
        with open(self.log_path, 'a+' ,encoding="utf-8") as f_txt:
            print(str(log),file = f_txt)
            print(log)
    
    def delete_file(self,file_or_dir):
        os.system(f"rm -rf {file_or_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,required=True,help="source data dir")
    parser.add_argument("--save_dir",type=str,required=True,help="save horizon_format dir")
    parser.add_argument("--log_path",type=str,required=True,help="save log path")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    converter = RearToHorizon(args.data_dir, args.save_dir, args.log_path)
    converter.do_convert()

# run 指令
# python3  /workspace/cap-sh/tools/dataset_converters/rear2horizon_format.py \
#     --data_dir /workspace/data/changan_data/rear_detection/zhoushi/DTS000000891 \
#     --save_dir /workspace/data/changan_data/rear_detection/zhoushi_horizon_format/horizon_format_891 \
#     --log_path /workspace/data/changan_data/rear_detection/zhoushi/DTS000000891.txt