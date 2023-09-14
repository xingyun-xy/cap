"""Build kitti2d."""

import argparse
import os

from cap.data.datasets.kitti2d import Kitti2DDetectionPacker
from cap.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Pack kitti2d dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The target pack type for result of packer",
    )
    parser.add_argument(
        "--target-data-dir",
        default="./data/kitti2d",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--anno-file",
        required=True,
        help="The path of kitti_train.json or kitti_eval.json",
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    init_logger(".cap_logs/kitti_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir
    split_name = os.path.basename(args.anno_file).split(".")[0]

    pack_path = os.path.join(
        args.target_data_dir,
        "%s_%s" % (split_name, args.pack_type),
    )
    packer = Kitti2DDetectionPacker(
        directory,
        pack_path,
        args.anno_file,
        int(args.num_workers),
        args.pack_type,
        None,
    )
    packer()
