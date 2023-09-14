import argparse
import json

import numpy as np
import os
import fnmatch
import shutil

classmap = {
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
occlusion_map = {
    '完全可见':'full_visible',
    '部分遮挡':'occluded',
    '严重遮挡':'heavily_occluded',
    '完全不可见':'invisible',
    '遮挡':'occluded',
    "图外":"outside" #based on the document of labeling guide, changan dataset contains this, also automatrix labels contains 'self_occluded' attribute.
}

confidence_map = {
    '高':'High',
    '中':'Middle',
    '低':'Low',
    '非常低':'VeryLow' 
}

truncation_map = {
    '非截断':'None',
    '有效部分高':'High',
    '有效部分中':'Middle',
    '有效部分低':'Low',
    '有效部分非常低':'VeryLow'
}
ignore_map = {
    "是":"yes",
    "否":"no"
}
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert voc annotations to COCO format"
    )
    parser.add_argument(
        "-p",
        "--json-path",
        type=str,
        required=True,
        help="original data path",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="anno.json",
        help="output path",
    )
    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        default="ext",
        help="extra name",
    )
    return parser.parse_args()


class LabelToDensebox(object):
    def __init__(self, data_dir, save_path, ext):
        self.annotations = []
        self.data_dir = data_dir
        self.save_path = save_path
        self.folder_name = ext + data_dir[data_dir.rfind('/')+1:]
        self.all_json_files = self.get_all_json_files(self.data_dir)
        self.img_folder = self.get_img_folder(self.data_dir)

    def get_all_json_files(self, data_dir):
        json_files_path = list()
        for dir_path, dir_name, files in os.walk(data_dir):
            for f in fnmatch.filter(files, '*.json'):
                json_files_path.append(os.path.join(dir_path, f))

        return json_files_path

    def get_img_folder(self, data_dir):
        image_folder = data_dir.replace('annotation/','')
        return image_folder

    def json_to_densebox(self):
        json_list = self.all_json_files
        for json_path in json_list:
            bbox_id = 1
            kps_id = 300001
            denseboxes = []
            f = open(json_path, "r",encoding = 'utf-8')
            image_folder = json_path[json_path.find('/') + 1:]
            print(image_folder)
            label = list(map(lambda x: json.loads(x,strict=False), f.readlines()))[0]['markData']
           
            width = label['width']
            height = label['height']           
            #automatrix kps label contains image_uuid keyword dont know changan data contains uuid but found uid(unsure if it is uuid or not)
            # uid =label['annotations'][0]['data'][0]['nlp'][0]['uid']
            uid = -1
            image_src = label['src']
            image_key = image_src[image_src.rfind('/')+1:]
            annotations = label['annotations']
            
            # print(annotations)
            image_attrs = {}

            belongs_to =[]
            kps_front_rear = []
            vehicle_kps_8 =[]
            tag = label["tagsItems"]

               
            for anno in annotations:
                densebox = {}
                attrs = {}
                kpsall = {}
                vehicle8 = {}
                if anno['data'] == []:
                    print('anno_data:',anno['data'])
                    continue
                anno_data = anno['data'][0]
                if len(anno)>3:
                    if "question" in anno:
                        kpsattr = list(anno['question'].values())
                    if kpsattr == []:
                        print('kpsattr = [] ', image_src)
                        break
                    bl = f"vehicle|{bbox_id}:vehicle_kps_8|{kps_id}"
                    belongs_to.append(bl)
                    kps_rear = anno['data'][1]
                    kps_front = anno['data'][2]
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
                    
                    vehicle8["point_attrs"] = [{"point_label":{"Corner_confidence": Corner_confidence[kpsattr[2]], "ignore":ignore_map[kpsattr[1]], "occlusion":occlusion_map[kpsattr[0]], "position":"other"}},
                    {"point_label":{"Corner_confidence":Corner_confidence[kpsattr[-1]], "ignore":ignore_map[kpsattr[-2]], "occlusion":occlusion_map[kpsattr[-3]], "position":"other"}}]
                    vehicle8["struct_type"] ="vehicle_kps_8"
                    vehicle8["track_id"] = -1
                    vehicle_kps_8.append(vehicle8)
                    kps_id+=1   
                    # vehicle8["point_attrs"] = 
                    # kpsall["vehicle_kps_8"]
                if 'question' not in anno_data:
                    print('json_path:',json_path)
                    continue
                ques_list = list(anno_data['question'].values())
                
                data = anno_data['relativePos']
             
                image_attrs["belong_to"] = belongs_to
                densebox["id"] = bbox_id
                densebox["track_id"] = -1
                densebox["data"] = [data['left'],data['top'],data['right'],data['bottom']]
                densebox["struct_type"] = anno_data['type']
                attrs["type"] = classmap[ques_list[-1]]
                print(attrs["type"])
                print("\n")
                attrs["ignore"] = ignore_map[ques_list[1]]
                attrs["occlusion"] = occlusion_map[ques_list[0]]
                attrs["confidence"] = confidence_map[ques_list[2]]
                attrs["truncation"] = truncation_map[ques_list[-2]]
                densebox["attrs"] = attrs
                bbox_id = bbox_id + 1     
                # kps_id+=1          
                # import pdb
                # pdb.set_trace()
                denseboxes.append(densebox)

            if kpsattr == []:
                continue
            image_attrs["height"] = height
            image_attrs["image_key"] = image_key
            image_attrs["image_source"]=""
            image_attrs["image_uuid"] = uid 
            image_attrs["video_name"] = "1"
            image_attrs["video_index"] = "1"
            image_attrs["vehicle"] = denseboxes
            image_attrs["vehicle_kps_8"] = vehicle_kps_8
            image_attrs["width"] = width
            self.annotations.append(image_attrs)

    def save_imgs(self):
        target_path = self.save_path + '/' + self.folder_name + 'data/'
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        src_files = os.listdir(self.img_folder)
        for file_name in src_files:
            full_file_name = os.path.join(self.img_folder, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, target_path)

    def save_json(self):
        self.json_to_densebox()
        json_path = self.save_path + '/' + self.folder_name + 'data.json'
        with open(json_path, 'w') as f:
            for anno in self.annotations:
                json_str = json.dumps(anno)
                f.write(json_str+'\n')

def Car2dToDensebox(json_path, out_dir, ext):
    json_path_list = os.listdir(json_path)
    # print(json_path_list)
    for json_pth in json_path_list:
        full_json_pth = json_path + json_pth
        print(full_json_pth)
        c = LabelToDensebox(full_json_pth, out_dir, ext)
        c.save_json()
        c.save_imgs()


if __name__ == "__main__":
    # args = parse_args()
    # json_path = r'/data/changan_data/5004/DTS000000623+安驿-2D图像-全车-正式-周视-2325/dataset-623/version-1/annotation/'
    # json_path = r'/data/changan_data/5004/DTS000000636+安驿-2D图像-全车-正式-周视-20135/dataset-636/version-1/annotation/'
    # json_path = r'/data/changan_data/5004/DTS000000638+安驿-2D图像-全车-正式-周视-24755/dataset-638/version-1/annotation/周视/'
    # json_path = r'/data/changan_data/5017/DTS000000654+安驿-2D图像-全车-正式/dataset-654/version-1/annotation/周视/'
    # json_path = r'/data/changan_data/5017/DTS000000654+安驿-2D图像-全车-正式/dataset-729/version-1/annotation/'
    # json_path = r'/data/changan_data/5017/DTS000000730+安驿-2D图像-全车-正式-周视-250/dataset-730/version-1/annotation/'
    # json_path = r'/data/xfwang/data/test/pilot_test/annotation/'
    # json_path = r'C:/Users/202209606/Desktop/v2/annotation/'
    # json_path = r'/data/temp/changan_data/vehicle_keypoints/zhoushi/DTS000000971/version-11/annotation/'
    # out_dir = r'/data/temp/changan_data/vehicle_keypoints/zhoushi_horizon_format/horizon_format_971/version-11'
    json_path = r'/data/temp/changan_data/vehicle_keypoints/zhoushi/DTS000001225/'
    out_dir = r'/data/temp/changan_data/vehicle_keypoints/zhoushi_horizon_format/'
    
    
    ext = 'densebox_'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # Car2dToDensebox(json_path, out_dir, ext)
    dirs = os.listdir(json_path)
    print('dirs = ', dirs)
    for cur_dir in dirs:
        print('cur_dir = ', cur_dir)
        cur_out_dir = out_dir + 'horizon_format_' + json_path[-4:-1] + '/' + cur_dir
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        cur_json_path = json_path + cur_dir + '/annotation/'
        Car2dToDensebox(cur_json_path, cur_out_dir, ext)

