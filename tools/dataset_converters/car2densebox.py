import argparse
import json

import numpy as np
import os
import fnmatch
import shutil

classmap = {
    '轿车':'Sedan_Car',
    '越野':'SUV',
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

occlusion_map = {
    '完全可见':'full_visible',
    '部分遮挡(1~30%)':'occluded',
    '严重遮挡(31~70%)':'heavily_occluded',
    '完全不可见(>70%)':'invisible'
}

confidence_map = {
    '高':'High',
    '中':'Middle',
    '低':'Low',
    '非常低':'VeryLow'
}

truncation_map = {
    '非截断':'None',
    '可见有效部分高':'High',
    '有效部分中':'Middle',
    '有效部分低':'Low',
    '有效部分非常低':'VeryLow'
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
            denseboxes = []
            f = open(json_path, "r")
            image_folder = json_path[json_path.find('/') + 1:]
            label = list(map(lambda x: json.loads(x), f.readlines()))[0]['markData']
            width = label['width']
            height = label['height']
            image_src = label['src']
            image_key = image_src[image_src.rfind('/')+1:]
            annotations = label['annotations']
            image_attrs = {}
            image_attrs["image_key"] = image_key
            image_attrs["video_name"] = "1"
            image_attrs["video_index"] = "1"
            image_attrs["width"] = width
            image_attrs["height"] = height

            for anno in annotations:
                densebox = {}
                attrs = {}
                anno_data = anno['data'][0]
                if 'question' not in anno_data:
                    print('json_path:',json_path)
                    continue
                ques_list = list(anno_data['question'].values())
                data = anno_data['relativePos']
                densebox["id"] = bbox_id
                densebox["track_id"] = -1
                densebox["data"] = [data['left'],data['top'],data['right'],data['bottom']]
                densebox["struct_type"] = anno_data['type']
                attrs["type"] = classmap[ques_list[0]]
                attrs["ignore"] = ques_list[1]
                attrs["occlusion"] = occlusion_map[ques_list[2]]
                attrs["confidence"] = confidence_map[ques_list[3]]
                attrs["truncation"] = truncation_map[ques_list[4]]
                densebox["attrs"] = attrs
                bbox_id = bbox_id + 1
                denseboxes.append(densebox)
            image_attrs["vehicle"] = denseboxes
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

    for json_pth in json_path_list:
        full_json_pth = json_path + json_pth
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
    json_path = r'/data/temp/changan_data/vehicle_detection/zhoushi/DTS000000990/version-2/annotation/'
    out_dir = r'/data/temp/changan_data/vehicle_detection/zhoushi/horizon_format_990/version-2/'
    
    ext = 'densebox_'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    Car2dToDensebox(json_path, out_dir, ext)
