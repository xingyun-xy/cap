"""viz cityscapes."""

import argparse
import pickle

from cap.data.datasets import FlyingChairs
from cap.utils.logger import init_logger
from cap.visualize import FlowViz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
    )
    parser.add_argument(
        "--viz-num",
        default=50,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )

    args = parser.parse_args()

    init_logger(".cap_logs/flow_viz")

    dataset = FlyingChairs(
        data_path=args.data_path,
        to_rgb=True,
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = FlowViz(is_plot=args.plot)
    for i, data in enumerate(dataset):

        img = data["img"]
        gt_flow = data["gt_flow"]

        if args.plot:
            viz(img, gt_flow)
        if i > int(args.viz_num):
            break
