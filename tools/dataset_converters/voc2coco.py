"""This script describes how to convert the voc dataset into a COCO format dataset.

You can refer to this script to convert your dataset into COCO format.
Reference: https://github.com/open-mmlab/mmdetection/blob/master/tools/dataset_converters/pascal_voc.py
"""  # noqa: E501

import argparse
import json

import numpy as np

from cap.data.datasets.voc import _PASCAL_VOC_LABELS, PascalVOC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert voc annotations to COCO format"
    )
    parser.add_argument(
        "-p",
        "--pack-data-path",
        type=str,
        required=True,
        help="voc data path",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        default="pascal_voc.json",
        help="output path",
    )
    return parser.parse_args()


def set_categories():
    categories = []
    for name, category_id in _PASCAL_VOC_LABELS.items():
        category_item = {}
        category_item["supercategory"] = str(category_id[1])
        category_item["id"] = int(category_id[0])
        category_item["name"] = str(name)
        categories.append(category_item)
    return categories


def cvt_to_coco_json(pack_data_path, out_json_name):
    annotation_id = 0
    coco = {}
    coco["images"] = []
    coco["categories"] = []
    coco["annotations"] = []
    image_set = set()

    coco["categories"] = set_categories()
    voc_dataset = PascalVOC(
        data_path=pack_data_path,
        transforms=None,
    )

    print(f"There are a total of {len(voc_dataset)} images")
    for i in range(len(voc_dataset)):
        data = voc_dataset[i]
        image_item = {}
        image_id = int(data["img_id"][0])
        image_name = str("%012d" % image_id + ".jpg")
        assert image_name not in image_set
        image_item["id"] = image_id
        image_item["file_name"] = image_name
        image_item["height"] = data["img_height"]
        image_item["width"] = data["img_width"]
        image_set.add(image_name)
        coco["images"].append(image_item)

        gt_bboxes = data["gt_bboxes"]
        gt_classes = data["gt_classes"]
        gt_difficult = data["gt_difficult"]
        for j in range(gt_difficult.shape[0]):
            annotation_item = {}
            annotation_item["segmentation"] = []
            seg = []
            bbox = gt_bboxes[j]
            # left_top
            seg.append(int(bbox[0]))
            seg.append(int(bbox[1]))
            # left_bottom
            seg.append(int(bbox[0]))
            seg.append(int(bbox[3]))
            # right_bottom
            seg.append(int(bbox[2]))
            seg.append(int(bbox[3]))
            # right_top
            seg.append(int(bbox[2]))
            seg.append(int(bbox[1]))

            annotation_item["segmentation"].append(seg)
            xywh = np.array(
                [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            )
            annotation_item["area"] = int(xywh[2] * xywh[3])

            if int(gt_difficult[j]) == 1:
                annotation_item["ignore"] = 0
                annotation_item["iscrowd"] = 1
            else:
                annotation_item["ignore"] = 0
                annotation_item["iscrowd"] = 0
            annotation_item["image_id"] = image_id
            annotation_item["bbox"] = xywh.astype(int).tolist()
            annotation_item["category_id"] = int(gt_classes[j])
            annotation_item["id"] = int(annotation_id)
            coco["annotations"].append(annotation_item)
            annotation_id += 1

    with open(out_json_name, "w") as f:
        json.dump(coco, f)


if __name__ == "__main__":
    args = parse_args()
    cvt_to_coco_json(args.pack_data_path, args.out_dir)
