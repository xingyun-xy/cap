"""viz cityscapes."""

import argparse
import pickle

import torchvision

from cap.data.datasets import Cityscapes
from cap.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS
from cap.data.transforms import LabelRemap, PILToTensor
from cap.utils.logger import init_logger
from cap.visualize import SegViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--viz-num",
        default=5000,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".cap_logs/cityscapes_viz")

    dataset = Cityscapes(
        data_path=args.data_path,
        transforms=torchvision.transforms.Compose(
            [
                PILToTensor(),
                LabelRemap(mapping=CITYSCAPES_LABLE_MAPPINGS),
            ]
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = SegViz()

    for i, data in enumerate(dataset):
        img = data["img"]
        gt_seg = data["gt_seg"]
        if args.plot:
            viz(img, gt_seg)
        if i > int(args.viz_num):
            break
