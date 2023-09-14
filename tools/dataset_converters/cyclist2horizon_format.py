# 查看文件夹内文件个数
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
    '人骑骑摩托车及电动车': 'PersonRideMotorcycle',
    '骑摩托车及电动车的人': 'Motorcyclist',
    '人骑自行车': 'PersonRideBicycle',
    '骑自行车的人': 'Bicyclist',
}

OCCLUSION_MAP = {
    '完全可见': 'full_visible',
    '部分遮挡': 'occluded',
    '严重遮挡': 'heavily_occluded',
    '不可见': 'invisible',
}

CONFIDENCE_MAP = {
    '高': 'High',
    '中': 'Middle',
    '低': 'Low',
    '非常低': 'VeryLow'
}

IGNORE_MAP = {
    '是': 'yes',
    '否': 'no',
}

OVERLAPPEDBOX_MAP = {
    '否': 'no',
    '是': 'yes'
}

POINT_OCCLUSION_MAP = {
    '完全可见': 'full_visible',
    '遮挡': 'occluded',
    '图外': 'occluded',
}

POINT_CONFIDENCE_MAP = {
    '高': 'high',
    '中': 'middle',
    '低': 'low',
    '非常低': 'super_low'
}

POINT_IGNORE_MAP = {
    '是': 'yes',
    '否': 'no',
}


class CyclistToHorizon(object):
    def __init__(self, data_dir, save_path, log_path):
        self.data_dir = data_dir
        self.data_id = data_dir.split("/")[-1]
        self.save_path = save_path
        self.log_path = log_path

        self.content_error_json_path = []
        self.question_error_json_path = []
        self.data_error_json_path = []
        self.kps_error_json_path = []
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
                    with open(json_path, "r", encoding='utf-8') as f:
                        try:  # 捕获json 内容错误
                            label = json.load(f)
                        except Exception as e:
                            self.content_error_json_path.append(json_path)
                            self.save_json_flag = False
                            continue
                        new_label = {}
                        new_label["image_key"] = image_name
                        new_label["image_source"] = " "
                        new_label["video_name"] = "1"
                        new_label["video_index"] = "1"
                        new_label["width"] = label["markData"]['width']
                        new_label["height"] = label["markData"]['height']
                        new_label["person"] = []
                        new_label["belong_to"] = []
                        new_label["p_WheelKeyPoints_2"] = []
                        for i, item in enumerate(label["markData"]['annotations']):
                            anno = {}
                            anno["id"] = i + 1
                            anno["track_id"] = -1
                            anno["label_type"] = "boxes"
                            anno["struct_type"] = "rect"
                            anno["attrs"] = {}
                            try:  # 捕获question为空的json
                                question_dict = item["data"][0]["question"]
                                anno["attrs"]["type"] = CLASS_MAP[question_dict["117"]]
                                anno["attrs"]["confidence"] = CONFIDENCE_MAP[question_dict["120"]]
                                anno["attrs"]["OverlappedBox"] = OVERLAPPEDBOX_MAP[question_dict["121"]]
                                # anno["attrs"]["age"] = "Adult"  # 骑车人数据不用于行人年龄分类任务
                                anno["attrs"]["ignore"] = IGNORE_MAP[question_dict["118"]]
                                anno["attrs"]["occlusion"] = OCCLUSION_MAP[question_dict["119"]]
                            except Exception as e:
                                self.question_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break
                            anno["data"] = []
                            try:  # 捕获data为空的json # 解决"relativePos": [] 这个bug
                                assert item['data'][0]['relativePos']
                                bbox_data = item['data'][0]['relativePos']
                                left = round(bbox_data['left'], 3)
                                top = round(bbox_data['top'], 3)
                                right = round(bbox_data['right'], 3)
                                bottom = round(bbox_data['bottom'], 3)
                                anno["data"] = [left, top, right, bottom]
                            except Exception as e:
                                self.data_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break
                            new_label["person"].append(anno)

                            # kps标注转换
                            try:
                                kps = {}
                                kps['data'] = [list(map(lambda x: round(x, 3), repos['relativePos'].values())) for repos in item["data"][1:]]
                                assert all(kps['data']) and len(kps['data']) == 2
                                
                                kps['id'] = f'30{i + 1:04d}'
                                kps['luid'] = ' '
                                kps['track_id'] = -1
                                kps['num'] = 2
                                kps['struct_type'] = "p_WheelKeyPoints_2"
                                kps['label_type'] = "points"
                                kps['attrs'] = {'type': anno['attrs']['occlusion']}

                                point_att_rear = {}
                                p_question = item.pop('question')
                                point_att_rear['occlusion'] = POINT_OCCLUSION_MAP[p_question['70']]
                                point_att_rear['ignore'] = POINT_IGNORE_MAP[p_question['71']]
                                point_att_rear['Corner_confidence'] = POINT_CONFIDENCE_MAP[p_question['72']]
                                point_att_front = {}
                                point_att_front['occlusion'] = POINT_OCCLUSION_MAP[p_question['73']]
                                point_att_front['ignore'] = POINT_IGNORE_MAP[p_question['74']]
                                point_att_front['Corner_confidence'] = POINT_CONFIDENCE_MAP[p_question['75']]
                                kps['point_attrs'] = [point_att_rear,point_att_front] 

                                new_label["belong_to"].append(f"person|{anno['id']}:p_WheelKeyPoints_2|{kps['id']}")
                                new_label['p_WheelKeyPoints_2'].append(kps)

                            except Exception:
                                self.kps_error_json_path.append(json_path)
                                self.save_json_flag = False
                                break

                    if self.save_json_flag:
                        if self.is_contain_chinese(image_name):
                            self.contain_chinese_json_path.append(json_path)
                        else:
                            self.effective_json_path.append(json_path)
                            self.save_json(target_path, new_label)  # save json
                            shutil.copy(image_path, target_data_path)  # copy image

        total_error_json_nums = len(self.content_error_json_path) + \
                                len(self.question_error_json_path) + \
                                len(self.data_error_json_path) + \
                                len(self.kps_error_json_path) + \
                                len(self.repeat_image_name_list) + \
                                len(self.contain_chinese_json_path)

        log = f"=========================数据集:{self.data_id}========================= \
            \n====>存在content问题的json_path:{self.content_error_json_path} \
            \n====>存在question问题的json_path:{self.question_error_json_path} \
            \n====>存在data问题的json_path:{self.data_error_json_path} \
            \n====>存在kps问题的json_path:{self.kps_error_json_path} \
            \n====>image_key重复的json_path:{self.repeat_json_path} \
            \n====>image_key包含中文的json_path:{self.contain_chinese_json_path} \
            \n================================================================== \
            \n====>json文件总数:{self.total_json_nums} \
            \n====>有问题json文件总数:{total_error_json_nums} \
            \n     ====>其中json文件content错误总数:{len(self.content_error_json_path)} \
            \n     ====>其中json文件question错误总数:{len(self.question_error_json_path)} \
            \n     ====>其中json文件data错误总数:{len(self.data_error_json_path)} \
            \n     ====>其中json文件kps错误总数:{len(self.kps_error_json_path)} \
            \n     ====>其中json文件repeat总数:{len(self.repeat_image_name_list)} \
            \n     ====>其中json文件image_key包含中文错误总数:{len(self.contain_chinese_json_path)} \
            \n====>有效的json文件总数:{len(self.effective_json_path)} \
            \n====>json文件合格率:{1 - total_error_json_nums / self.total_json_nums} \
            \n===============================done!=============================="
        self.log2txt(log)

    def is_contain_chinese(self, check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def get_file_info(self, json_path):
        version_info = json_path.split("/annotation/")[0].split("/")[-1]
        cam_info = json_path.split("/")[-2]
        target_path = os.path.join(self.save_path, self.data_id + "-" + version_info + "-" + cam_info)
        target_data_path = os.path.join(target_path, "data")
        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)
        image_path = json_path.replace("annotation/", "").replace(".json", ".jpg")
        image_name = image_path.split("/")[-1]
        assert os.path.exists(image_path), "image_path不存在!"
        return image_path, target_path, target_data_path, image_name, version_info

    def save_json(self, target_path, new_label):
        json_path = target_path + '/data.json'
        with open(json_path, 'a+') as f:
            json_str = json.dumps(new_label)
            f.write(json_str + '\n')

    def log2txt(self, log):
        with open(self.log_path, 'a+', encoding="utf-8") as f_txt:
            print(str(log), file=f_txt)
            print(log)

    def delete_file(self, file_or_dir):
        os.system(f"rm -rf {file_or_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="source data dir")
    parser.add_argument("--save_dir", type=str, required=True, help="save horizon_format dir")
    parser.add_argument("--log_path", type=str, required=True, help="save log path")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    converter = CyclistToHorizon(args.data_dir, args.save_dir, args.log_path)
    converter.do_convert()

# run 指令
# python3  cyclist2horizon_format \
#     --data_dir /data/changan_data/cyclist_keypoints/qianshi_duan/DTS000000949/ \
#     --save_dir /data//changan_data/cyclist_keypoints/qianshi_duan_horizon_format/DTS000000949/ \
#     --log_path /data/changan_data/cyclist_keypoints/qianshi_duan/DTS000000949.txt
