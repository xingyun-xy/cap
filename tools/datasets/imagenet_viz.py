"""viz imagenet."""

import argparse
import pickle

import torch

from cap.data.datasets import ImageNet
from cap.utils.logger import init_logger
from cap.visualize import ClsViz

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

    init_logger(".cap_logs/imagenet_viz")

    dataset = ImageNet(
        data_path=args.data_path,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = ClsViz(is_plot=args.plot)

    for i, data in enumerate(dataset):
        img = data["img"].permute(1, 2, 0).numpy()
        label = torch.from_numpy(data["labels"]).unsqueeze(0)
        viz(img, label)
        if i > int(args.viz_num):
            break
