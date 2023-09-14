# LaneSegmentation & Default Segmentation
## mean_iou: 类别平均IOU
## mean_acc: 类别平均准确率
## all_acc: 所有类别的准确率
## 各个类别的iou和acc

## 混淆矩阵

## 创建评测集
### 目录结构

/LaneSegmentation  
/LaneSegmentation/data  
/LaneSegmentation/gt  
/LaneSegmentation/data.json 
$\underline{changanformat}$

/DefaultSegmentation  
/DefaultSegmentation/data  
/DefaultSegmentation/gt  
/DefaultSegmentation/data.json  $\underline{changanformat}$

# Real3D evluation

## real3d评测集结构
参考路径: /data/eval/nuscenes_eval/

| 目录                   | 说明                            |
| ---------------------- | -----------------              |
| data                   |                                |
| ----xxx.jpg  ...       | 所有图片                        |
| data.json              | gt的json，与打包时的json格式相同 |

## real3d评测指标
"dx": x方向位置绝对误差,  
"dy": y方向位置绝对误差,  
"dxy": 位置绝对误差,  
"dw": 目标宽W的绝对误差,  
"dl": 目标长L的绝对误差,  
"dxp": x方向位置百分比误差,  
"dyp": y方向位置百分比误差,  
"dxyp": 位置百分比误差,  
"dwp": 目标宽W的百分比误差,  
"dlp": 目标长L的百分比误差,  
"drot": 朝向角绝对误差,  
