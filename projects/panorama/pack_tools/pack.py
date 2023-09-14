import argparse

from packer import LmdbDefaultPacker

from cap.data.packer.utils import set_pack_env
from cap.utils import Config


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="configure file"
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=16,
        help="the number of worker to pack data",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()

    set_pack_env(
        num_worker=args.num_worker,
        verbose=args.verbose,
    )

    return args


def setup_config(args):
    config = Config.fromfile(args.config)
    return config


def main():

    args = parse_args()
    config = setup_config(args)
    logger = config.logger
    logger.info("=" * 50 + "BEGIN PACKING %s" + "=" * 50, config.task_name)

    packer = LmdbDefaultPacker(config)
    packer.pack_idx()
    packer.pack_anno()
    packer.pack_img()
    if args.visualize:
        packer.visualize()

    logger.info("=" * 50 + "END PACKING %s" + "=" * 50, config.task_name)


if __name__ == "__main__":
    main()