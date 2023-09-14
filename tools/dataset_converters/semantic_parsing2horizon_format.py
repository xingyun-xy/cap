# 产看文件夹内文件个数
# ls -lR|grep "^-"|wc -l

#cam1  左前
#cam2  左后
#cam3  右前
#cam4  右后
#cam5  正后

#0-result 右前 对应cam3
#1-result 右后 对应cam4
#2-result 左前 对应cam1
#3-result 左后 对应cam2
#4-result 正后 对应cam5


import os
import json
import shutil
import argparse
from glob import glob  # 获取全局图片
from tqdm import tqdm


CLASS_MAP = {
    '车道':'road',
    '人行道':'sidewalk',
    '植被':'vegetation',
    '地形':'terrain',
    '杆子':'pole',
    '交通标志牌':'traffic_sign',
    '交通灯':'traffic_light',
    '标志线':'Sign_Line',
    '车道线':'lane_marking',
    '人':'person',
    '骑行者':'rider',
    '自行车':'bicycle',
    '摩托车':'motorcycle',
    '三轮车':'tricycle',
    '小汽车':'car',
    '卡车':'truck',
    '公交车':'bus',
    '火车':'train',
    '建筑物':'building',
    '围栏':'fence',
    '天空':'sky',
    '路锥':'Traffic_Cone',
    '防护柱':'Bollard',
    '指路牌':'Guide_Post',
    '斑马线':'Crosswalk_Line',
    '箭头':'Traffic_Arrow',
    '导流线':'Guide_Line',
    '停止线':'Stop_Line',
    '三角路标':'Slow_Down_Triangle',
    '限速路标':'Speed_Sign',
    '菱形':'Diamond',
    '自行车路标':'Bicyclesign',
    '减速带':'SpeedBumps',
    '禁止通行标志':'no_forward_marker',
    '停车杆':'parking_rod',
    '停车锁':'parking_lock',
    '可跨越障碍物':'traversable_obstruction',
    '不可跨越障碍物':'untraversable_obstruction',
    '掩码':'mask',
    '其他':'other',
    '立柱':'parking_column',
    '禁停线':'no_parking_line',
    '减速让行线':'slow_down_line',
    '地面文字':'road_text',
    '停止让行线':'stop_attention_line',
    '收费杆':'toll_pole',
}


class SemanticParsingToHorizon(object):
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
                    image_path,target_path,target_data_path,image_name,version_info = self.get_file_info(json_path)
                    # process repeat
                    if image_name in self.effective_json_path: 
                        self.repeat_image_name_list.append(image_name)
                        self.repeat_json_path.append(json_path)
                        self.save_json_flag = False
                        continue
                    #process json
                    with open(json_path,"r",encoding='utf-8') as f:
                        try: # 捕获json 内容错误
                            label = json.load(f)
                        except Exception as e:
                            self.content_error_json_path.append(json_path)
                            self.save_json_flag = False
                            continue
                        new_label = {}
                        new_label["image_key"] = image_name
                        new_label["video_name"] = "1"
                        new_label["video_index"] = "1"
                        new_label["width"] = label["markData"]['width']
                        new_label["height"] = label["markData"]['height']
                        new_label["parsing"] = []
                        for i, item in enumerate(label["markData"]['annotations']):
                            anno = {}
                            anno["id"] = i
                            anno["track_id"] = -1
                            anno["struct_type"] = "parsing"
                            anno["attrs"] = {}
                            try:  #捕获question为空的json
                                anno["attrs"]["type"] = CLASS_MAP[item["title"]]
                                anno["attrs"]["ignore"] = item["question"]["32"]
                            except Exception as e:
                                self.question_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break

                            anno["data"] = []
                            try:  #捕获data为空的json
                                assert item['data'][0]['relativePos']#解决"relativePos": [] 这个bug,或者是空的 "data": []
                                for xy_pos in item['data'][0]['relativePos']:
                                    anno["data"].append([format(xy_pos["x"], '.3f'), format(xy_pos["y"], '.3f')])
                            except Exception as e:
                                self.data_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break

                            anno["point_attrs"] = [None]*len(anno["data"])
                            new_label["parsing"].append(anno)
                    
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
        version_info = json_path.split("/annotation/")[0].split("/")[-1]
        cam_info = json_path.split("/")[-2]
        target_path = os.path.join(self.save_path, self.data_id + "-" + version_info + "-" + cam_info)
        target_data_path = os.path.join(target_path,"data")
        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)
        image_path =  json_path.replace("annotation/", "").replace(".json",".jpg")
        image_name = image_path.split("/")[-1]
        assert os.path.exists(image_path),"image_path不存在!"
        return image_path,target_path,target_data_path,image_name,version_info

    def save_json(self,target_path,new_label):
        json_path = target_path + '/data.json'
        with open(json_path, 'a+') as f:
            json_str = json.dumps(new_label)
            f.write(json_str+'\n')

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
    converter = SemanticParsingToHorizon(args.data_dir, args.save_dir, args.log_path)
    converter.do_convert()
