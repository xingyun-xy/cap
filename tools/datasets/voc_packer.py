"""Pack original voc."""

import argparse
import os

from cap.data.datasets.voc import VOCDetectionPacker
from cap.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Pack voc dataset.")
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
        default="",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="trainval", help="The split to pack."
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

    init_logger(".cap_logs/voc_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir

    pack_path = os.path.join(
        args.target_data_dir,
        "%s_%s" % (args.split_name, args.pack_type),
    )
    packer = VOCDetectionPacker(
        directory,
        pack_path,
        args.split_name,
        int(args.num_workers),
        args.pack_type,
        None,
    )
    packer()
