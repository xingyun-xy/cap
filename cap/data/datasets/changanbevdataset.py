import os
import glob
import mmcv
import numpy as np
import torch
from PIL import Image
from cap.data.datasets.bevdepth import Quaternion
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import cv2
import math
from cap.registry import OBJECT_REGISTRY
__all__ = ['changanbevdataset']

use_other_pre = False

def get_rot(h):
    return torch.Tensor([
        [np.cos(h) , np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_transform(img, resize, resize_dims, crop, flip, rotate):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)
    # adjust image
    # img = img.resize(resize_dims)
    # img = img.crop(crop)
    # if flip:
    #     img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    # img = img.rotate(rotate)
    img = cv2.resize(img, tuple(resize_dims))  # 改用opencv进行读取
    img = img[crop[1]:crop[3],crop[0]:crop[2],:]
    if flip:
        img = cv2.flip(img,1)
    img = np.asarray(Image.fromarray(img).rotate(rotate))

    # post-homography transformation
    ida_rot[0,0] *= resize[0] # 这里的x和y单独进行处理
    ida_rot[1,1] *= resize[1]
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat

@OBJECT_REGISTRY.register
class changanbevdataset(Dataset):

    def __init__(self,
                 pre_process_info, # 初始化时所需要的信息
                 ):
        
        super().__init__()
        self.pre_process_info = pre_process_info
        # print(pre_process_info)
        self.classes          = pre_process_info['CLASS'] # 类别名字
        self.data_root        = pre_process_info["data_root"]# 数据路径  
        self.img_mean         = np.array(pre_process_info["img_conf"]['img_mean'], np.float32) # 均值
        self.img_std          = np.array(pre_process_info["img_conf"]['img_std'], np.float32) # 标准差
        self.img_avg_mean     = np.array([np.mean(self.img_mean),np.mean(self.img_mean),np.mean(self.img_mean)]) # 平均均值 图像增广的时候可以考虑使用
        self.img_avg_std      = np.array([np.mean(self.img_std),np.mean(self.img_std),np.mean(self.img_std)]) # 平均标准差 图像增广的时候可以考虑使用
        
        self.to_rgb           = pre_process_info["img_conf"]['to_rgb'] # 是否要进行通道转换 默认为False，直接按照cv2的BGR进行处理
        self.visual_imgs      = pre_process_info["visual_imgs"] # 对处理后的图像进行可视化
        if self.visual_imgs:
            self.visual_save_path = pre_process_info["visual_save_path"] # 处理后的图像可视化保存的路径
        self.train_flag = pre_process_info["train_flag"] # 是否是训练的标志
        
        self.cams   = self.choose_cams()     # 选择6个camera ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.params = self.get_all_camIE()   # 拿到内外参数以及畸变矫正系数
        # 按照顺序读取内参
        self.intrin_mats = torch.stack([torch.from_numpy(np.array(self.params["Camera_FLC"]["camera_intrinsic"])), torch.from_numpy(np.array(self.params["Camera_FC"]["camera_intrinsic"])), torch.from_numpy(np.array(self.params["Camera_FRC"]["camera_intrinsic"])),
                                        torch.from_numpy(np.array(self.params["Camera_RLC"]["camera_intrinsic"])), torch.from_numpy(np.array(self.params["Camera_RC"]["camera_intrinsic"])), torch.from_numpy(np.array(self.params["Camera_RRC"]["camera_intrinsic"]))])\
                                           .view(6, 4, 4)
        # 按照顺序读取外参
        self.sensor2ego_mats = torch.stack([torch.from_numpy(np.array(self.params["Camera_FLC"]["camera_extern"]['4x4'])), torch.from_numpy(np.array(self.params["Camera_FC"]["camera_extern"]['4x4'])), torch.from_numpy(np.array(self.params["Camera_FRC"]["camera_extern"]['4x4'])),
                                            torch.from_numpy(np.array(self.params["Camera_RLC"]["camera_extern"]['4x4'])), torch.from_numpy(np.array(self.params["Camera_RC"]["camera_extern"]['4x4'])), torch.from_numpy(np.array(self.params["Camera_RRC"]["camera_extern"]['4x4']))])\
                                                .view(6, 4, 4)
        # 畸变矫正系数
        self.distort = {'Camera_FLC':self.params["Camera_FLC"]["distort"], 'Camera_FC': self.params["Camera_FC"]["distort"], 'Camera_FRC': self.params["Camera_FRC"]["distort"],
                        'Camera_RLC':self.params["Camera_RLC"]["distort"], 'Camera_RC': self.params["Camera_RC"]["distort"], 'Camera_RRC': self.params["Camera_RRC"]["distort"],}
        
        self.sensor2ego_rot = torch.stack([torch.from_numpy(np.array(self.params["Camera_FLC"]["camera_extern"]['rotation']['quat'])), torch.from_numpy(np.array(self.params["Camera_FC"]["camera_extern"]['rotation']['quat'])), torch.from_numpy(np.array(self.params["Camera_FRC"]["camera_extern"]['rotation']['quat'])),
                                           torch.from_numpy(np.array(self.params["Camera_RLC"]["camera_extern"]['rotation']['quat'])), torch.from_numpy(np.array(self.params["Camera_RC"]["camera_extern"]['rotation']['quat'])), torch.from_numpy(np.array(self.params["Camera_RRC"]["camera_extern"]['rotation']['quat']))])\
                                                .view(6, 4)
        self.sensor2ego_trans = torch.stack([torch.from_numpy(np.array(self.params["Camera_FLC"]["camera_extern"]['trans'])), torch.from_numpy(np.array(self.params["Camera_FC"]["camera_extern"]['trans'])), torch.from_numpy(np.array(self.params["Camera_FRC"]["camera_extern"]['trans'])),
                                             torch.from_numpy(np.array(self.params["Camera_RLC"]["camera_extern"]['trans'])), torch.from_numpy(np.array(self.params["Camera_RC"]["camera_extern"]['trans'])), torch.from_numpy(np.array(self.params["Camera_RRC"]["camera_extern"]['trans']))])\
                                                .view(6, 3)
        # 要求每个视角下的图像名字都是一样的 这里是读取图像
        self.img_names_CAM_FRONT_LEFT,self.img_names_CAM_FRONT,self.img_names_CAM_FRONT_RIGHT,\
        self.img_names_CAM_BACK_LEFT ,self.img_names_CAM_BACK ,self.img_names_CAM_BACK_RIGHT = [],[],[],[],[],[]
        for names in glob.glob(self.data_root + "/CAM_FRONT_LEFT/*.jpg"): # 暂时默认图像是jpg格式的 要求图像放在固定的目录下，名字都一样
            self.img_names_CAM_FRONT_LEFT.append(names)
            self.img_names_CAM_FRONT.append(names.replace("/CAM_FRONT_LEFT/", "/CAM_FRONT/"))
            self.img_names_CAM_FRONT_RIGHT.append(names.replace("/CAM_FRONT_LEFT/", "/CAM_FRONT_RIGHT/"))
            self.img_names_CAM_BACK_LEFT.append(names.replace("/CAM_FRONT_LEFT/", "/CAM_BACK_LEFT/"))
            self.img_names_CAM_BACK.append(names.replace("/CAM_FRONT_LEFT/", "/CAM_BACK/"))
            self.img_names_CAM_BACK_RIGHT.append(names.replace("/CAM_FRONT_LEFT/", "/CAM_BACK_RIGHT/"))

        self.train = pre_process_info["train_flag"] # 是否是训练的标志
        self.return_undistort_imgs = pre_process_info["return_undistort_imgs"] # 是否返回畸变处理后的图像

    def eulerAngles2rotationMat(self, theta, format='degree'):
        """
        Calculates Rotation Matrix given euler angles.
        :param theta: 1-by-3 list [rx, ry, rz] angle in degree
        :return:
        RPY角, 是ZYX欧拉角, 依次绕定轴XYZ转动[rx, ry, rz]
        """
        # if format == 'degree':
        #     theta = [i * math.pi / 180.0 for i in theta]

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R
    
    def extend_trans_mat(self, rotationMat, translation):
        trans_mat = np.zeros((4, 4))
        trans_mat[:3, :3] = rotationMat
        trans_mat[:3, 3] = translation
        trans_mat[3, 3] = 1
        return trans_mat
    
    def get_all_camIE(self): # 获取内外餐和畸变矫正系数
        CAMS = ['Camera_FLC', 'Camera_FC', 'Camera_FRC', 'Camera_RLC', 'Camera_RC', 'Camera_RRC']

        # roll pitch yaw from calibration file
        Camera_FLC_eulerAngles = [0.006092571187764406, -0.015221013687551022, 0.9438098669052124]
        Camera_FC_eulerAngles = [0.01640610210597515, 0.01361369900405407, 0.023038338869810104]
        Camera_FRC_eulerAngles = [0.018006347119808197, 0.0241914801299572, -1.007336974143982]
        Camera_RLC_eulerAngles = [-0.029371898621320724, -0.01930270344018936, 2.2105422019958496]
        Camera_RC_eulerAngles = [-0.010977690108120441, -0.12539778649806976, 3.1101696491241455]
        Camera_RRC_eulerAngles = [0.039584480226039886, -0.0532054603099823, -2.2449522018432617]

        Camera_FLC_rotationMat = self.eulerAngles2rotationMat(Camera_FLC_eulerAngles)
        Camera_FC_rotationMat = self.eulerAngles2rotationMat(Camera_FC_eulerAngles)
        Camera_FRC_rotationMat = self.eulerAngles2rotationMat(Camera_FRC_eulerAngles)
        Camera_RLC_rotationMat = self.eulerAngles2rotationMat(Camera_RLC_eulerAngles)
        Camera_RC_rotationMat = self.eulerAngles2rotationMat(Camera_RC_eulerAngles)
        Camera_RRC_rotationMat = self.eulerAngles2rotationMat(Camera_RRC_eulerAngles)

        # translation from calibration file
        Camera_FLC_translation = [2.25, 0.9300000071525574, 0.0]
        Camera_FC_translation = [1.8200000524520874, -0.04500000178813934, 0.0]
        Camera_FRC_translation = [2.240000009536743, -1.0299999713897705, 0.0]
        Camera_RLC_translation = [2.5, 1.0099999904632568, 0.0]
        Camera_RC_translation = [-1.0399999618530273, 0.0, 0.0]
        Camera_RRC_translation = [2.509999990463257, -1.0299999713897705, 0.0]

        Camera_FLC = self.extend_trans_mat(Camera_FLC_rotationMat, Camera_FLC_translation)
        Camera_FC = self.extend_trans_mat(Camera_FC_rotationMat, Camera_FC_translation)
        Camera_FRC = self.extend_trans_mat(Camera_FRC_rotationMat, Camera_FRC_translation)
        Camera_RLC = self.extend_trans_mat(Camera_RLC_rotationMat, Camera_RLC_translation)
        Camera_RC = self.extend_trans_mat(Camera_RC_rotationMat, Camera_RC_translation)
        Camera_RRC = self.extend_trans_mat(Camera_RRC_rotationMat, Camera_RRC_translation)
        
        # 刘少林使用后处理库导出来的camera到自车的外参
        # Camera_FLC = [0.825026095, -0.565042853, 0.00765307853, 2.21171021,
        #             0.565038383, 0.825059474, 0.00294130249, 0.953895152,
        #             -0.00797620602, 0.00189763121, 0.999966383, 0.934060931,
        #             0, 0, 0, 1]
        
        # Camera_FC = [0.999297678, -0.0156982187, -0.0340274051, 1.81999993,
        #             0.0151709625, 0.99976176, -0.0156982243, 0.0450000018,
        #             0.0342657343, 0.0151709681, 0.999297619, 1.30999994,
        #             0, 0, 0, 1]

        # Camera_FRC = [0.808217466, 0.588846087, 0.0067047067, 2.27147651,
        #             -0.588774145, 0.808234155, -0.0101332599, -0.977069139,
        #             -0.0113859028, 0.00424231822, 0.999926209, 0.952619076,
        #             0, 0, 0, 1]
    
        # Camera_RLC = [-0.863190353, -0.503190339, 0.0412558988, 2.49547958,
        #             0.502710938, -0.864174008, -0.0220262539, 1.01272094,
        #             0.0467356704, 0.00172694295, 0.998905778, 0.737797976,
        #             0, 0, 0, 1]
        
        # Camera_RC = [-0.999083102, 0.0331056975, 0.0271460805, -1.11547136,
        #             -0.0329513848, -0.999438286, 0.00611237902, 0.0218286347,
        #             0.0273331851, 0.00521227391, 0.999612749, 0.970588207,
        #             0, 0, 0, 1]
        
        # Camera_RRC = [-0.862666488, 0.505769253, -0.00200974615, 2.68763709,
        #               -0.505771756, -0.862647355, 0.00586289726, -0.828961909,
        #              0.00123157084, 0.00607419712, 0.999980807, 0.904729605,
        #                0, 0, 0, 1]
        
        IES = { # 用的都是最原始的图像内参 来自于json文件
            "Camera_FLC": {
                "camera_intrinsic": [[1133.6961669921875, 0.0, 965.549072265625, 0.0],
                                     [0.0, 1134.693603515625, 766.3670654296875, 0.0],
                                     [0.0, 0.0, 1.0, 0.0], 
                                     [0.0, 0.0, 0.0, 1.0]],
                "distort":np.array([-0.3611186146736145, 0.161410391330719, 0.000671005283948034, 0.0006341779953800142, -0.038743291050195694])
            },
            "Camera_FC": {
                "camera_intrinsic": [[1907.8199462890625,        0,           1903.8900146484375,  0], 
                                     [0,                 1907.6700439453125,  1065.3800048828125,  0], 
                                     [0,                         0,                   1         ,  0],
                                     [0,                         0,                   0         ,  1]],
                "distort":np.array([0.6399999856948853, -0.006899999920278788, 0.0006544502102769911, 0.0001176481819129549, -0.00570000009611249, 1.0039000511169434, 0.13189999759197235, -0.02019999921321869])
            },
            "Camera_FRC": {
                "camera_intrinsic": [[1141.7784423828125, 0.0, 980.0226440429688, 0.0],
                                     [0.0, 1143.9937744140625, 779.5892944335938, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]],
                "distort":np.array([-0.36598771810531616, 0.16775691509246826, -0.0011937993112951517, -6.458094867412001e-05, -0.04211336746811867])
            },
            "Camera_RLC": {
                "camera_intrinsic": [[1131.8980712890625, 0.0, 953.396728515625, 0.0],
                                     [0.0, 1133.15625, 764.8165893554688, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]],
                "distort":np.array([-0.36739909648895264, 0.1740468442440033, -7.580517558380961e-05, 0.0015015676617622375, -0.0446963869035244])
            },
            "Camera_RC": {
                "camera_intrinsic": [[1777.3565673828125, 0.0, 989.002685546875, 0.0],
                                     [0.0, 1770.4656982421875, 684.9658203125, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]],
                "distort":np.array([-0.3855231702327728, 0.12599718570709229, 0.004489528015255928, -0.003086293349042535, 0.07080148160457611])
            },
            "Camera_RRC": {
                "camera_intrinsic": [[1128.546875, 0.0, 957.18212890625, 0.0],
                                     [0.0, 1118.1455078125, 753.8971557617188, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]],
                "distort":np.array([-0.353847473859787, 0.15162406861782074, 0.0005773368175141513, 0.00012161301856394857, -0.03455384820699692])
            }}

        horizon_to_nusc_rot = np.array( # 地平线向nuscenes转换的变换矩阵
            [[ 0,  0, 1],
             [-1,  0, 0],
             [ 0, -1, 0]]
        )
        
        for cam in CAMS:
            tmp = np.array(eval(cam)).reshape(4, 4) # 拿到外参
            r = np.dot(tmp[:3, :3], horizon_to_nusc_rot) # 外参的旋转矩阵部分，进行坐标轴变换，变成和nuscenes一样
            if 'camera_extern' not in IES[cam]: # 外参一开始并不在IES里边，这里为对应camera设置键值对，用于装载外参（camera==>ego）
                IES[cam].setdefault('camera_extern', dict())
            IES[cam]['camera_extern']['rotation'] = dict() # 在外参这个键中右增加了一个字典
            IES[cam]['camera_extern']['rotation']['matrix'] = r.tolist()# 外参旋转部分的矩阵
            x, y, z, w = R.from_matrix(r).as_quat()
            IES[cam]['camera_extern']['rotation']['quat'] = [w, x, y, z]# 外参旋转部分的四元组
            IES[cam]['camera_extern']['trans'] = tmp[:3, 3].tolist()# 外参平移部分 3个数值
            mat_4x4 = np.zeros((4, 4))
            mat_4x4[:3, :3] = r
            mat_4x4[:3, 3] = tmp[:3, 3].tolist()
            mat_4x4[3, 3] = 1
            IES[cam]['camera_extern']['4x4'] = mat_4x4.tolist() # 外参部分的齐次矩阵
        return IES
     
    def sample_ida_augmentation(self, pos = None):
        '''
        'final_dim': (256, 704), # 最终输入网络的参数
        'FC_H': 2160, # 前视camera的高
        'FC_W': 3840, # 前视camera的宽
        'NOT_FC_H': 1536, # 非前视camera的高
        'NOT_FC_W': 1920, # 非前视camera的宽
        
        'FLC_FRC_RLC_RRC_RC_TOP_DOWN': (0.25, 0.35), # 前左 前右 后左 后 后右 上下的crop起始点范围 
        'FC_TOP_DOWN'                : (0.00, 0.15), # 前视 上下的crop起始点范围 
        'RLC_RRC_DROP'               : (0.10, 0.11), # 后左和后右因为需要把左边的自车crop掉，这个是左或者右需要丢掉的范围
        '''
        FC_H, FC_W         = self.pre_process_info['FC_H'], self.pre_process_info['FC_W']         #  前视camera的高和宽  2160 3840
        NOT_FC_H, NOT_FC_W = self.pre_process_info['NOT_FC_H'], self.pre_process_info['NOT_FC_W'] # 非前视camera的高和宽 1536 1920
        fH, fW             = self.pre_process_info['final_dim'] # 256, 704
        resize, resize_dims, crop, flip, rotate_ida = None,None,None,None,None # 初始化图像增广参数
        
        if self.train:
            if pos == 'FLC' or pos == 'FRC' or pos == 'RC': # 前左、前右和后 左右全保留，高度上进行随机选择
                resize      = max(fW / NOT_FC_W, fH / NOT_FC_H) # (704 / 1920, 256 / 1536) = (0.3666666666666667, 0.1666666666666667)
                resize_dims = (int(NOT_FC_W * resize), int(NOT_FC_H * resize)) # 704 563 resize的宽和高
                top_start   = int(np.random.uniform(self.pre_process_info['FLC_FRC_RLC_RRC_RC_TOP_DOWN']) * resize_dims[1]) # 在高度上寻找起始点
                crop        = (0, top_start, fW, top_start + fH) # [左边 上边 右边 下边]
                resize      = (resize,resize)
                
            elif pos == 'FC': # 前视 左右全保留，高度上进行随机选择                
                resize      = max(fW / FC_W, fH / FC_H) # (704 / 3840，256 / 2160) = (0.1833333333333333，0.1185185185185185)
                resize_dims = (int(FC_W * resize), int(FC_H * resize)) # 704 396 resize的宽和高
                top_start   = int(np.random.uniform(self.pre_process_info['FC_TOP_DOWN']) * resize_dims[1]) # 在高度上寻找起始点
                crop        = (0, top_start, fW, top_start + fH) # [左边 上边 右边 下边]
                resize      = (resize,resize)
                
            elif pos == 'RLC': # 后左 需要的resize是两个数字 只有后左和后右需要在宽度上resize 宽度上也要进行crop
                # 计算仅需要进行上下crop的camera会把高度首先resize到多少 ==> 563
                resize         = max(fW / NOT_FC_W, fH / NOT_FC_H) # (704 / 1920, 256 / 1536) = (0.3666666666666667, 0.1666666666666667)
                resize_dims    = (int(NOT_FC_W * resize), int(NOT_FC_H * resize)) # 704 563
                top_start      = int(np.random.uniform(self.pre_process_info['FLC_FRC_RLC_RRC_RC_TOP_DOWN']) * resize_dims[1]) # 上边的起始点
                left_start     = np.random.uniform(self.pre_process_info['RLC_RRC_DROP']) # 左边丢掉比例，因为车身被拍到了，那部分不需要
                resize_dims[0] = math.ceil(resize_dims[0] / (1-left_start)) # 左边丢掉那么多时，还需要保证宽度有需要的宽度resize_dims[0]，故丢掉之前的宽度需要先计算出来
                left_start     = int(left_start * resize_dims[0]) # 左边丢掉的具体像素
                crop           = (left_start, top_start, left_start + fW, top_start + fH) # [左边 上边 右边 下边]
                resize         = (resize_dims[0] / NOT_FC_W, resize_dims[1] / NOT_FC_H)
                
            elif pos == 'RRC': # 后右 需要的resize是两个数字 只有后左和后右需要在宽度上resize 宽度上也要进行crop
                # 计算仅需要进行上下crop的camera会把高度首先resize到多少 ==> 563
                resize         = max(fW / NOT_FC_W, fH / NOT_FC_H) # (704 / 1920, 256 / 1536) = (0.3666666666666667, 0.1666666666666667)
                resize_dims    = (int(NOT_FC_W * resize), int(NOT_FC_H * resize)) # 704 563
                top_start      = int(np.random.uniform(self.pre_process_info['FLC_FRC_RLC_RRC_RC_TOP_DOWN']) * resize_dims[1])# 上边的起始点
                crop           = (0, top_start, resize_dims[0], top_start + fH) # [左边 上边 右边 下边]
                
                right_end      = np.random.uniform(self.pre_process_info['RLC_RRC_DROP']) # 右边丢掉多少
                resize_dims[0] = math.ceil(resize_dims[0] / (1-right_end))# 右边丢掉那么多时，还需要保证宽度有需要的宽度resize_dims[0]，故丢掉之前的宽度需要先计算出来
                right_end      = int(right_end * resize_dims[0]) # 右边丢掉的具体像素
                resize         = (resize_dims[0] / NOT_FC_W, resize_dims[1] / NOT_FC_H)
            
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.ida_aug_conf['rot_lim'])
            
        else: ################# 目前仅仅调试这里
            if pos == 'FLC' or pos == 'FRC' or pos == 'RC' or pos == 'RLC' or pos == 'RRC': # 前左、前右和后 左右全保留，高度上进行随机选择 
                resize      = max(fW / NOT_FC_W, fH / NOT_FC_H) # (704 / 1920, 256 / 1536) = (0.3666666666666667, 0.1666666666666667)
                resize_dims = (int(NOT_FC_W * resize), int(NOT_FC_H * resize)) # 704 563
                top_start   = int(np.mean(self.pre_process_info['FLC_FRC_RLC_RRC_RC_TOP_DOWN']) * resize_dims[1])
                crop        = (0, top_start, fW, top_start + fH) # [左边 上边 右边 下边]
                resize      = (resize, resize)

                if use_other_pre:
                    resize      = (fW / NOT_FC_W, fH / NOT_FC_H) 
                    resize_dims = (576, 320)
                    crop        = (0, 0, fW, fH) 
            elif pos == 'FC': # 前视 左右全保留，高度上进行随机选择
                resize      = max(fW / FC_W, fH / FC_H) # (704 / 3840，256 / 2160) = (0.1833333333333333，0.1185185185185185)
                resize_dims = (int(FC_W * resize), int(FC_H * resize)) # 704 396
                top_start   = int(np.mean(self.pre_process_info['FC_TOP_DOWN']) * resize_dims[1])
                crop        = (0, top_start, fW, top_start + fH) # [左边 上边 右边 下边]
                resize      = (resize,resize)

                if use_other_pre:
                    resize      = (fW / FC_W, fH / FC_H)
                    resize_dims = (576, 320)
                    crop        = (0, 0, fW, fH) 
            # elif pos == 'RLC': # 后左 需要的resize是两个数字 只有后左和后右需要在宽度上resize 宽度上也要进行crop
            #     # 计算仅需要进行上下crop的camera会把高度首先resize到多少 ==> 563
            #     resize         = max(fW / NOT_FC_W, fH / NOT_FC_H) # (704 / 1920, 256 / 1536) = (0.3666666666666667, 0.1666666666666667)
            #     resize_dims    = [int(NOT_FC_W * resize), int(NOT_FC_H * resize)] # 704 563
            #     top_start      = int(np.mean(self.pre_process_info['FLC_FRC_RLC_RRC_RC_TOP_DOWN']) * resize_dims[1]) # 上边的起始点
            #     left_start     = np.mean(self.pre_process_info['RLC_RRC_DROP']) # 左边丢掉比例
            #     resize_dims[0] = math.ceil(resize_dims[0] / (1-left_start)) # 左边丢掉那么多时，需要的宽度
            #     left_start     = int(left_start * resize_dims[0]) # 左边丢掉的具体像素
            #     crop           = (left_start, top_start, left_start + fW, top_start + fH) # [左边 上边 右边 下边]
            #     resize         = (resize_dims[0] / NOT_FC_W, resize_dims[1] / NOT_FC_H)
                
            # elif pos == 'RRC': # 后右 需要的resize是两个数字 只有后左和后右需要在宽度上resize 宽度上也要进行crop
            #     # 计算仅需要进行上下crop的camera会把高度首先resize到多少 ==> 563
            #     resize         = max(fW / NOT_FC_W, fH / NOT_FC_H) # (704 / 1920, 256 / 1536) = (0.3666666666666667, 0.1666666666666667)
            #     resize_dims    = [int(NOT_FC_W * resize), int(NOT_FC_H * resize)] # 704 563
            #     top_start      = int(np.mean(self.pre_process_info['FLC_FRC_RLC_RRC_RC_TOP_DOWN']) * resize_dims[1])# 上边的起始点
            #     crop           = (0, top_start, resize_dims[0], top_start + fH) # [左边 上边 右边 下边]
                
            #     right_end      = np.mean(self.pre_process_info['RLC_RRC_DROP']) # 右边丢掉多少
            #     resize_dims[0] = math.ceil(resize_dims[0] / (1-right_end))# 右边丢掉那么多时，需要的宽度
            #     right_end      = int(right_end * resize_dims[0]) # 右边丢掉的具体像素
            #     resize         = (resize_dims[0] / NOT_FC_W, resize_dims[1] / NOT_FC_H)
                
            flip = False
            rotate_ida = 0
            
        return resize, resize_dims, crop, flip, rotate_ida

    def choose_cams(self):
        """Choose cameras randomly.

        Returns:
            list: Cameras to be used.
        """
        cams = self.pre_process_info['cams']
        return cams
       
    def __getitem__(self, idx):
        
        # 这些仅仅产生图像增广，不处理图像
        resize_CAM_FRONT_LEFT , resize_dims_CAM_FRONT_LEFT , crop_CAM_FRONT_LEFT  , flip_CAM_FRONT_LEFT , rotate_ida_CAM_FRONT_LEFT = self.sample_ida_augmentation("FLC")
        resize_CAM_FRONT      , resize_dims_CAM_FRONT      , crop_CAM_FRONT       , flip_CAM_FRONT      , rotate_ida_CAM_FRONT      = self.sample_ida_augmentation("FC")
        resize_CAM_FRONT_RIGHT, resize_dims_CAM_FRONT_RIGHT, crop_CAM_FRONT_RIGHT , flip_CAM_FRONT_RIGHT, rotate_ida_CAM_FRONT_RIGHT= self.sample_ida_augmentation("FRC")
        resize_CAM_BACK_LEFT  , resize_dims_CAM_BACK_LEFT  , crop_CAM_BACK_LEFT   , flip_CAM_BACK_LEFT  , rotate_ida_CAM_BACK_LEFT  = self.sample_ida_augmentation("RLC")
        resize_CAM_BACK       , resize_dims_CAM_BACK       , crop_CAM_BACK        , flip_CAM_BACK       , rotate_ida_CAM_BACK       = self.sample_ida_augmentation("RC")
        resize_CAM_BACK_RIGHT , resize_dims_CAM_BACK_RIGHT , crop_CAM_BACK_RIGHT  , flip_CAM_BACK_RIGHT , rotate_ida_CAM_BACK_RIGHT = self.sample_ida_augmentation("RRC")
        # 读取 6 摄像图像 这里需要矫正畸变 
        # img_CAM_FRONT_LEFT_UNDIS = cv2.undistort(cv2.imread(self.img_names_CAM_FRONT_LEFT[idx]) , self.intrin_mats[0].numpy()[0:3,0:3], self.distort['Camera_FLC'], None, self.intrin_mats[0].numpy()[0:3,0:3]) 
        # img_CAM_FRONT_UNDIS      = cv2.undistort(cv2.imread(self.img_names_CAM_FRONT[idx])      , self.intrin_mats[1].numpy()[0:3,0:3], self.distort['Camera_FC'] , None, self.intrin_mats[1].numpy()[0:3,0:3])
        # img_CAM_FRONT_RIGHT_UNDIS= cv2.undistort(cv2.imread(self.img_names_CAM_FRONT_RIGHT[idx]), self.intrin_mats[2].numpy()[0:3,0:3], self.distort['Camera_FRC'], None, self.intrin_mats[2].numpy()[0:3,0:3])
        # img_CAM_BACK_LEFT_UNDIS  = cv2.undistort(cv2.imread(self.img_names_CAM_BACK_LEFT[idx])  , self.intrin_mats[3].numpy()[0:3,0:3], self.distort['Camera_RLC'], None, self.intrin_mats[3].numpy()[0:3,0:3])
        # img_CAM_BACK_UNDIS       = cv2.undistort(cv2.imread(self.img_names_CAM_BACK[idx])       , self.intrin_mats[4].numpy()[0:3,0:3], self.distort['Camera_RC'] , None, self.intrin_mats[4].numpy()[0:3,0:3])
        # img_CAM_BACK_RIGHT_UNDIS = cv2.undistort(cv2.imread(self.img_names_CAM_BACK_RIGHT[idx]) , self.intrin_mats[5].numpy()[0:3,0:3], self.distort['Camera_RRC'], None, self.intrin_mats[5].numpy()[0:3,0:3])
        img_CAM_FRONT_LEFT_UNDIS = cv2.imread(self.img_names_CAM_FRONT_LEFT[idx])
        img_CAM_FRONT_UNDIS = cv2.imread(self.img_names_CAM_FRONT[idx])
        img_CAM_FRONT_RIGHT_UNDIS = cv2.imread(self.img_names_CAM_FRONT_RIGHT[idx])
        img_CAM_BACK_LEFT_UNDIS = cv2.imread(self.img_names_CAM_BACK_LEFT[idx])
        img_CAM_BACK_UNDIS = cv2.imread(self.img_names_CAM_BACK[idx])
        img_CAM_BACK_RIGHT_UNDIS = cv2.imread(self.img_names_CAM_BACK_RIGHT[idx])

        # 图像预处理
        img_CAM_FRONT_LEFT , ida_mat_CAM_FRONT_LEFT  = img_transform(img_CAM_FRONT_LEFT_UNDIS  , resize=resize_CAM_FRONT_LEFT  , resize_dims=resize_dims_CAM_FRONT_LEFT , crop=crop_CAM_FRONT_LEFT , flip=flip_CAM_FRONT_LEFT , rotate=rotate_ida_CAM_FRONT_LEFT ,)
        img_CAM_FRONT      , ida_mat_CAM_FRONT       = img_transform(img_CAM_FRONT_UNDIS       , resize=resize_CAM_FRONT       , resize_dims=resize_dims_CAM_FRONT      , crop=crop_CAM_FRONT      , flip=flip_CAM_FRONT      , rotate=rotate_ida_CAM_FRONT      ,)
        img_CAM_FRONT_RIGHT, ida_mat_CAM_FRONT_RIGHT = img_transform(img_CAM_FRONT_RIGHT_UNDIS , resize=resize_CAM_FRONT_RIGHT , resize_dims=resize_dims_CAM_FRONT_RIGHT, crop=crop_CAM_FRONT_RIGHT, flip=flip_CAM_FRONT_RIGHT, rotate=rotate_ida_CAM_FRONT_RIGHT,)
        img_CAM_BACK_LEFT  , ida_mat_CAM_BACK_LEFT   = img_transform(img_CAM_BACK_LEFT_UNDIS   , resize=resize_CAM_BACK_LEFT   , resize_dims=resize_dims_CAM_BACK_LEFT  , crop=crop_CAM_BACK_LEFT  , flip=flip_CAM_BACK_LEFT  , rotate=rotate_ida_CAM_BACK_LEFT  ,)
        img_CAM_BACK       , ida_mat_CAM_BACK        = img_transform(img_CAM_BACK_UNDIS        , resize=resize_CAM_BACK        , resize_dims=resize_dims_CAM_BACK       , crop=crop_CAM_BACK       , flip=flip_CAM_BACK       , rotate=rotate_ida_CAM_BACK       ,)
        img_CAM_BACK_RIGHT , ida_mat_CAM_BACK_RIGHT  = img_transform(img_CAM_BACK_RIGHT_UNDIS  , resize=resize_CAM_BACK_RIGHT  , resize_dims=resize_dims_CAM_BACK_RIGHT , crop=crop_CAM_BACK_RIGHT , flip=flip_CAM_BACK_RIGHT , rotate=rotate_ida_CAM_BACK_RIGHT ,)
        # 图像进行normalization
        # img_CAM_FRONT_LEFT = mmcv.imnormalize(np.array(img_CAM_FRONT_LEFT) , self.img_mean, self.img_std, self.to_rgb)
        # img_CAM_FRONT      = mmcv.imnormalize(np.array(img_CAM_FRONT)      , self.img_mean, self.img_std, self.to_rgb)
        # img_CAM_FRONT_RIGHT= mmcv.imnormalize(np.array(img_CAM_FRONT_RIGHT), self.img_mean, self.img_std, self.to_rgb)
        # img_CAM_BACK_LEFT  = mmcv.imnormalize(np.array(img_CAM_BACK_LEFT)  , self.img_mean, self.img_std, self.to_rgb)
        # img_CAM_BACK       = mmcv.imnormalize(np.array(img_CAM_BACK)       , self.img_mean, self.img_std, self.to_rgb)
        # img_CAM_BACK_RIGHT = mmcv.imnormalize(np.array(img_CAM_BACK_RIGHT) , self.img_mean, self.img_std, self.to_rgb)
        
        # 可视化保存
        if self.visual_imgs: ############## 如果要进行可视化，则在这里写入文件
            '''
            说明：后处理使用Image读取图像，读出来的通道顺序是rgb，但是使用mmcv.normalize进行处理后，
            转了图像通道，所以最终使用的图像是bgr的通道，而我可视化的时候本来就用bgr，所以反向normalize时，不需要重新转通道
            
            预处理之后的图像宽度：  704宽度 | 10间隔 | 704宽度 | 10间隔 | 704宽度
            
            预处理之后的图像高度：  256高度
                                  -------
                                  20间隔
                                  -------
                                  256高度
            '''
            print("***********************************************************************************")
            os.makedirs(self.visual_save_path, exist_ok=True)
            
            width_interval = 10
            height_interval = 20
            width = self.pre_process_info['final_dim'][1]
            height = self.pre_process_info['final_dim'][0]
            ######### 写预处理后的图像
            post_big_img = np.zeros((height + height_interval + height,  width + width_interval  + width + width_interval + width , 3)) + 128
            
            post_big_img[0:height, 0:width, :] = mmcv.imdenormalize(img_CAM_FRONT_LEFT, self.img_mean, self.img_std, to_bgr = False)
            post_big_img[0:height, width+width_interval:2*width+width_interval, :] = mmcv.imdenormalize(img_CAM_FRONT, self.img_mean, self.img_std, to_bgr = False)
            post_big_img[0:height, 2*width+2*width_interval:3*width+2*width_interval, :] = mmcv.imdenormalize(img_CAM_FRONT_RIGHT, self.img_mean, self.img_std, to_bgr = False)
            post_big_img[height+height_interval:, 0:width, :] = mmcv.imdenormalize(img_CAM_BACK_LEFT, self.img_mean, self.img_std, to_bgr = False)
            post_big_img[height+height_interval:, width+width_interval:2*width+width_interval, :] = mmcv.imdenormalize(img_CAM_BACK, self.img_mean, self.img_std, to_bgr = False)
            post_big_img[height+height_interval:, 2*width+2*width_interval:3*width+2*width_interval, :] = mmcv.imdenormalize(img_CAM_BACK_RIGHT, self.img_mean, self.img_std, to_bgr = False)
            
            save_path = self.visual_save_path + os.sep + os.path.basename(self.img_names_CAM_FRONT_LEFT[idx])
            save_path = save_path[0:save_path.rfind(".")] + "_post" + save_path[save_path.rfind("."):]
            cv2.imwrite(save_path, post_big_img.astype(np.uint8))
            
            ######### 写原始的图像
            H, W = self.pre_process_info['NOT_FC_H'], self.pre_process_info['NOT_FC_W'] # 非前视的高和宽
            fH, fW = self.pre_process_info['final_dim']
            resize = max(fH / H, fW / W) # (256 / 1536, 704 / 1920) = (0.1666666, 0.366666666)
            width = int(self.pre_process_info['NOT_FC_W'] * resize)
            height= int(self.pre_process_info['NOT_FC_H'] * resize)
            
            post_big_img = np.zeros((height + height_interval + height,  width + width_interval  + width + width_interval + width, 3)) + 128

            post_big_img[0:height, 0:width, :] = cv2.resize(img_CAM_FRONT_LEFT_UNDIS,(width, height))
            post_big_img[0:height, width+width_interval:2*width+width_interval, :] = cv2.resize(img_CAM_FRONT_UNDIS,(width, height))
            post_big_img[0:height, 2*width+2*width_interval:3*width+2*width_interval, :] = cv2.resize(img_CAM_FRONT_RIGHT_UNDIS,(width, height))
            post_big_img[height+height_interval:, 0:width, :] = cv2.resize(img_CAM_BACK_LEFT_UNDIS,(width, height))
            post_big_img[height+height_interval:, width+width_interval:2*width+width_interval, :] = cv2.resize(img_CAM_BACK_UNDIS,(width, height))
            post_big_img[height+height_interval:,  2*width+2*width_interval:3*width+2*width_interval, :] = cv2.resize(img_CAM_BACK_RIGHT_UNDIS,(width, height))
            
            save_path = self.visual_save_path + os.sep + os.path.basename(self.img_names_CAM_FRONT_LEFT[idx])
            save_path = save_path[0:save_path.rfind(".")] + "_ori" + save_path[save_path.rfind("."):]
            cv2.imwrite(save_path, post_big_img.astype(np.uint8))
        
        # 把通道维度挪动到前面
        img_CAM_FRONT_LEFT = torch.from_numpy(img_CAM_FRONT_LEFT).permute(2, 0, 1)
        img_CAM_FRONT      = torch.from_numpy(img_CAM_FRONT).permute(2, 0, 1)
        img_CAM_FRONT_RIGHT= torch.from_numpy(img_CAM_FRONT_RIGHT).permute(2, 0, 1)
        img_CAM_BACK_LEFT  = torch.from_numpy(img_CAM_BACK_LEFT).permute(2, 0, 1)
        img_CAM_BACK       = torch.from_numpy(img_CAM_BACK).permute(2, 0, 1)
        img_CAM_BACK_RIGHT = torch.from_numpy(img_CAM_BACK_RIGHT).permute(2, 0, 1)
        
        # 预处理后图像列表
        imgs = [img_CAM_FRONT_LEFT, img_CAM_FRONT, img_CAM_FRONT_RIGHT, img_CAM_BACK_LEFT, img_CAM_BACK, img_CAM_BACK_RIGHT]
        imgs = torch.stack(imgs).view(1, 6, 3, self.pre_process_info['final_dim'][0], self.pre_process_info['final_dim'][1])
        
        # 预处理后的图像增广的单应性矩阵
        ida_mats = [ida_mat_CAM_FRONT_LEFT, ida_mat_CAM_FRONT, ida_mat_CAM_FRONT_RIGHT, ida_mat_CAM_BACK_LEFT, ida_mat_CAM_BACK, ida_mat_CAM_BACK_RIGHT]
        ida_mats = torch.stack(ida_mats).view(1, 6, 4, 4) # 这个数据增广的矩阵是随着预处理而改变的，所以这里不能写死
        
        # bev空间的增广的对应矩阵
        bda_mat  = torch.from_numpy(np.eye(4)).to(self.sensor2ego_mats) # 里不是训练，所以直接写死，等到了需要训练的时候，这块重新组织代码逻辑
        
        # 返回预处理后的图像、camera到自车的转换外参、camera内参、图像增广对应的单应性矩阵、bev空间增广对应的矩阵、六摄图像的路径（FLC、FC、FRC、RLC、RC、RRC）
        
        # self.img_names_CAM_FRONT_LEFT[idx], self.img_names_CAM_FRONT[idx], self.img_names_CAM_FRONT_RIGHT[idx], \
        # self.img_names_CAM_BACK_LEFT[idx] , self.img_names_CAM_BACK[idx] , self.img_names_CAM_BACK_RIGHT[idx]
        if self.return_undistort_imgs:
            data_dict = dict()
            # B,N,C,H,W =imgs.shape
            
            # img = imgs.view(B*N,C,H,W)
            data_list = [
                            imgs.to(torch.uint8),
                            self.sensor2ego_mats.to(torch.float32).view(1,6, 4, 4),
                            self.intrin_mats.to(torch.float32).view(1,6, 4, 4),
                            ida_mats.to(torch.float32),
                            bda_mat.to(torch.float32),
                            self.sensor2ego_mats.to(torch.float32),
                            self.sensor2ego_trans.to(torch.float32),
                            [self.img_names_CAM_FRONT_LEFT[idx], 
                            self.img_names_CAM_FRONT[idx], 
                            self.img_names_CAM_FRONT_RIGHT[idx],
                            self.img_names_CAM_BACK_LEFT[idx], 
                            self.img_names_CAM_BACK[idx], 
                            self.img_names_CAM_BACK_RIGHT[idx]],
                            [
                                img_CAM_FRONT_LEFT_UNDIS, 
                                img_CAM_FRONT_UNDIS, 
                                img_CAM_FRONT_RIGHT_UNDIS, 
                                img_CAM_BACK_LEFT_UNDIS, 
                                img_CAM_BACK_UNDIS, 
                                img_CAM_BACK_RIGHT_UNDIS,
                            ],
                            self.sensor2ego_rot.to(torch.float32),
                    ]
            return data_list
                #imgs.to(torch.float32), self.sensor2ego_mats.to(torch.float32), self.intrin_mats.to(torch.float32), ida_mats.to(torch.float32), bda_mat.to(torch.float32), self.sensor2ego_rot.to(torch.float32),self.sensor2ego_trans.to(torch.float32),\
                #img_CAM_FRONT_LEFT_UNDIS, img_CAM_FRONT_UNDIS, img_CAM_FRONT_RIGHT_UNDIS, img_CAM_BACK_LEFT_UNDIS, img_CAM_BACK_UNDIS, img_CAM_BACK_RIGHT_UNDIS,\
                #self.img_names_CAM_FRONT_LEFT[idx], self.img_names_CAM_FRONT[idx], self.img_names_CAM_FRONT_RIGHT[idx], \
                #self.img_names_CAM_BACK_LEFT[idx] , self.img_names_CAM_BACK[idx] , self.img_names_CAM_BACK_RIGHT[idx]
        else:
            # data_dict = dict()
            # # B,N,C,H,W =imgs.shape
            
            # # img = imgs.view(B*N,C,H,W)
            # data_dict["img"] = imgs.to(torch.uint8)
            # data_dict["sensor2ego_mats"] = self.sensor2ego_mats.to(torch.float32)
            # data_dict["intrin_mats"] = self.intrin_mats.to(torch.float32)
            # data_dict["ida_mats"] =ida_mats.to(torch.float32)
            # data_dict["bda_mat"] = bda_mat.to(torch.float32)
            # data_dict["sensor2sensor_mats"] = self.sensor2ego_mats.to(torch.float32)
            # # data_dict["sensor2ego_trans"] = self.sensor2ego_trans.to(torch.float32)
            # data_dict["img_metas_batch"] = [self.img_names_CAM_FRONT_LEFT[idx][0], 
            #                                 self.img_names_CAM_FRONT[idx][0], 
            #                                 self.img_names_CAM_FRONT_RIGHT[idx][0],
            #                                 self.img_names_CAM_BACK_LEFT[idx][0], 
            #                                 self.img_names_CAM_BACK[idx][0], 
            #                                 self.img_names_CAM_BACK_RIGHT[idx][0]]
            data_list = [
                            imgs.to(torch.uint8),
                            self.sensor2ego_mats.to(torch.float32).view(1,6, 4, 4),
                            self.intrin_mats.to(torch.float32).view(1,6, 4, 4),
                            ida_mats.to(torch.float32),
                            bda_mat.to(torch.float32),
                            self.sensor2ego_mats.to(torch.float32),
                            self.sensor2ego_trans.to(torch.float32),
                            [self.img_names_CAM_FRONT_LEFT[idx], 
                            self.img_names_CAM_FRONT[idx], 
                            self.img_names_CAM_FRONT_RIGHT[idx],
                            self.img_names_CAM_BACK_LEFT[idx], 
                            self.img_names_CAM_BACK[idx], 
                            self.img_names_CAM_BACK_RIGHT[idx]]
                    ]
            return data_list
            #return data_dict
            # return [imgs.to(torch.uint8), self.sensor2ego_mats.to(torch.float32), self.intrin_mats.to(torch.float32), ida_mats.to(torch.float32), bda_mat.to(torch.float32), self.sensor2ego_rot.to(torch.float32),self.sensor2ego_trans.to(torch.float32), \
            #      [self.img_names_CAM_FRONT_LEFT[idx], self.img_names_CAM_FRONT[idx], self.img_names_CAM_FRONT_RIGHT[idx], \
            #      self.img_names_CAM_BACK_LEFT[idx], self.img_names_CAM_BACK[idx], self.img_names_CAM_BACK_RIGHT[idx]]]
        
        
    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}. Augmentation Conf: {self.ida_aug_conf}"""

    def __len__(self):        
        return len(self.img_names_CAM_FRONT_LEFT)


if __name__ == "__main__":
    changan_pre_process_infos = {
        "train_flag":False, # 训练模式
        'rot_lim'   : (-5.4, 5.4), # 旋转度数 非弧度
        'rand_flip' : True, # 是否水平翻转
        'FLC_FRC_RLC_RRC_RC_TOP_DOWN': (0.00, 0.10), # 前左 前右 后左 后 后右 上下的crop起始点范围 
        'FC_TOP_DOWN'                : (0.00, 0.01), # 前视 上下的crop起始点范围 
        'RLC_RRC_DROP'               : (0.10, 0.11), # 后左和后右因为需要把自车部分crop掉，这个是左或者右需要丢掉的范围
        
        'final_dim' : (320, 576),  # 非后视最终输入网络的参数
        'FC_H'       : 2160,        # 前视camera的高
        'FC_W'       : 3840,        # 前视camera的宽
        'NOT_FC_H'   : 1536,        # 非前视camera的高
        'NOT_FC_W'   : 1920,        # 非前视camera的宽
        'cams'       : ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'], # camera信息 实际暂没有用到 数据的处理顺序是按照这个来的
        'Ncams'      : 6, # camera个数
        "img_conf"   : dict(img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=False),# 图像均值、标准差以及是否转通道顺序
        "CLASS"      :['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian','traffic_cone',], # 检测的类别
        "data_root"  : "../../../datas/cadata/changan_data",# 数据的路径
        "visual_imgs": True, #是否可视化的标志 会把图像保存在本地
        "visual_save_path" : '../changan_visual_path', # 如果可视化，则保存在这里
        "return_undistort_imgs":True # 返回输入网络之前的图像可视化，用于端到端可视化
}
    
    # val_dataset = changanbevdataset(changan_pre_process_infos)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=1,
    #     sampler=None,
    # )
    
    # for each in val_loader:

    #     imgs, sensor2ego, intrin, ida, bda, FLC_path, FC_path, FRC_path, RLC_path, RC_path, RRC_path = each
   