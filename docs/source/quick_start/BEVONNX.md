# ONNX CONVERSION BASED ON NEW REQUIREMENT BY ZWJ 05/23/2023/-05/24/2023
## 3 additional inputs added
### mlp_input
mlp_input = torch.rand((1,1,6,27))
### circle_map
circle_map = torch.rand((1,112,16384))
### ray_map
ray_map = torch.rand((1,216,16384))
# Files which were modified
## List of files:  
base_lss_fpn_matrixvt.py  
pred_multitask.py     
vismodels.py  
pred_singletask_bev.py  
singletask_bev.py  
collate.py
***
For more details please refer to the report which was posted in the wechat group if you are interested. The command of running the code is still the same as before.


