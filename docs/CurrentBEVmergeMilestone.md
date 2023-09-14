# A summary of BEV works under CAP from end of 03/2023to 06/02/2023 (Currently)
## 1
Backbone sharing between BEVDepth and other tasks, with huge modifications on ***bev_matrixvt.py, collates.py*** and ***base_lss_fpn_matrixvt.py***. Also, yuv444 conversion for BEVDepth was included. 
***
## 2
Inference and visulization of its results, new functional files added:***get_bboxes.py, visualize_nusc.py***. Modifications and add-ons can be found in ***det_multitask.py***:  
```python

    if "singletask_bev" in self.out_keys :
        result_bev = result["singletask_bev"] 
        # import numpy as np
        # np.save("preds.npy", result_bev[0][0])
        bv = BevBBoxes(result_bev)
        """Remove comment only when u want to visualize changan bev!!!! added by zwj"""
        # ret_list,img_metas, path= bv.changan_bev_visualize()
        # file_name = img_metas[0][0].split("/")[-1]
        # changan_visual(ret_list, batch["sensor2ego_trans"], batch["sensor2ego_rot"], batch["intrin_mats"], 
        #                path + file_name, batch["cameras_paths"], 
        #                return_undistort_imgs = True, score_threshold = 0.6, limit_range = 61.2)
        bev_f, all_img_metas, path, ret_list = bv.bev_visualize()
        # process ret_list for bev eval     add by zmj
        import torch
        import numpy as np
        for i in range(len(ret_list[0])):
            if isinstance(ret_list[0][i], torch.Tensor):
                ret_list[0][i] = ret_list[0][i].detach().cpu().numpy()
        ret_list[0][3].pop('box_type_3d')
        result_bev_list = list(result_bev)
        result_bev_list.append(ret_list)
        result["singletask_bev"] = tuple(result_bev_list)
        # bev_f,all_img_metas,path =bv.bev_visualize()
        # demo(bev_f,all_img_metas,path)
        demo(bev_f,all_img_metas,path)
        if len(self.out_keys) != 1:
            model_outs = [result[key] for key in self.out_keys if key != "singletask_bev"]
    else:
            model_outs = [result[key] for key in self.out_keys]
```
***
## 3
Inference can now use one ground truth file and in json format, modifications can be found in ***pred_singletask_bev.py*** and ***bevdepth.py***.
```python  
# when share dataloader with CAP original tasks，two places should be changed，keep rest the same   add by zmj
data_loader_bev_cooperate_pilot = deepcopy(data_loader_bev)
data_loader_bev_cooperate_pilot['dataset']['type'] = "NuscDetDatasetCooperatePilot"
data_loader_bev_cooperate_pilot['collate_fn'] = partial(collate_fn_bevdepth_cooperate_pilot, is_return_depth=False)
```
***
## 4
Changan data-set merge, new functional file added: ***changanbevdataset.py*** and ***pred_changanbev.py***. Modules were added in ***get_bboxes.py***, ***visual_nusc.py*** and ***collate.py***
```python
data_loader_changanbev = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        # type="NuscDetDataset",
        type="changanbevdataset",
        pre_process_info = changan_pre_process_infos
    ),
        collate_fn=partial(collate_fn_changanbev,
                       is_return_depth=False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        sampler=None,
)
```
***
## 5
Evaluation of BEV added, BEV performance on Nuscences data set can be checked.
***
## 6
ONNX convrsion based on new requirements.  
For more detailed information please refer to ***BEVONNX.md*** and the related report which was posted in the wechat group
***
## 7
base_lss_fpn_matrixvt.py modifications and codes cleaning.
For more detailed information please refer to ***BEVandBaselssfpnmatrixvt.md***.