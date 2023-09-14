# base_lss_fpn_matrixvt.py modifications by ZWJ From 05/29/2023-06/02/2023/
Originally, **depthnet, horiconv, Matrixvt and etc** were in one file.  
Due to the codes were lengthy, modifications and restructuring were carried out. As the result, **base_lss_fpn_matrixvt** now only contains two modules: **BaseLSSFPN_matrixvt and Matrixvt**, redundant codes (especially for ***if torch.onnx.is_in_onnx_export() else***) and comments (not all) were also removed in order to reduce the length and make the codes more neat.  
### Example:  
Before:
```python
if torch.onnx.is_in_onnx_export() or use_onnx: ########## 替换算子
            # tmp = ida_mat.view(batch_size * num_cams, ida_mat.size(2), ida_mat.size(3))
            # tmp = inverse_ac(tmp).unsqueeze(1)
            tmp = ida_mat.view(batch_size * num_cams, 1, ida_mat.size(2), ida_mat.size(3))
            points = points.view(-1,4,1)
            values = []
            for i in range(tmp.size(0)):
                values.append(tmp[i,:,:,:].expand(78848,4,4).matmul(points))
            points = torch.stack(values,dim=0).squeeze(-1)
            # points = ida_mat.matmul(points.unsqueeze(-1))
        else:
            points = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4).inverse().matmul(points.unsqueeze(-1))
        # cam_to_ego
        if torch.onnx.is_in_onnx_export() or use_onnx: ########## 替换算子
            points = torch.cat((points[:, :, :2] * points[:, :, 2:3], points[:, :, 2:]), 2)
            # points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)
        else:
            points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)
        
        # combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        if torch.onnx.is_in_onnx_export() or use_onnx: ########## 替换算子
            # tmp = intrin_mat.view(batch_size*num_cams, intrin_mat.size(2), intrin_mat.size(3))
            # combine = sensor2ego_mat.matmul(inverse_ac(tmp).view(batch_size, num_cams, intrin_mat.size(2), intrin_mat.size(3)))
            combine = sensor2ego_mat.matmul(intrin_mat)
        else:
            combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        if torch.onnx.is_in_onnx_export() or use_onnx:
            combine = combine.view(batch_size * num_cams, 1, 4, 4)
            # points = combine.view(batch_size * num_cams, 1, 4, 4).matmul(points.unsqueeze(-1))
            points = points.unsqueeze(-1)
            values = []
            for i in range(tmp.size(0)):
                values.append(combine[i,:,:,:].expand(78848,4,4).matmul(points[i]))
            points = torch.stack(values,dim=0)
            # points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        else:
            points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        
        if torch.onnx.is_in_onnx_export() or use_onnx:
            # import pdb
            # pdb.set_trace()
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            # points = points.squeeze(-1)
            points = points.view(batch_size, num_cams, 112, 16, 44, 4)
        else:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1) # 常规的矩阵相乘
        return points[..., :3]
```
After:
```python
if torch.onnx.is_in_onnx_export() or use_onnx: ########## replace operator
            tmp = ida_mat.view(batch_size * num_cams, 1, ida_mat.size(2), ida_mat.size(3))
            points = points.view(-1,4,1)
            values = []
            for i in range(tmp.size(0)):
                values.append(tmp[i,:,:,:].expand(78848,4,4).matmul(points))
            points = torch.stack(values,dim=0).squeeze(-1)
            points = torch.cat((points[:, :, :2] * points[:, :, 2:3], points[:, :, 2:]), 2)
            combine = sensor2ego_mat.matmul(intrin_mat)
            combine = combine.view(batch_size * num_cams, 1, 4, 4)
            points = points.unsqueeze(-1)
            values = []
            for i in range(tmp.size(0)):
                values.append(combine[i,:,:,:].expand(78848,4,4).matmul(points[i]))
            points = torch.stack(values,dim=0)
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = points.view(batch_size, num_cams, 112, 16, 44, 4)
        else:
            points = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4).inverse().matmul(points.unsqueeze(-1))
            points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)
            combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
            points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points) 
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1) # Regular matrix multiplication
        return points[..., :3]
```  
Besides, everything else was moved to a new file which is called **bevmatrixvtcommon.py**. 
Also, base_lss_fpn_matrixvt can not be considered as a part of backbone, therefore, these two files (**base_lss_fpn_matrixvt.py** and **bevmatrixvtcommon.py**) were relocated at **./cap/models/task_modules/bev**.