# noqa: D205,D400
"""This script helps to mount the bucket and create a soft-link
of the bucket to the CAP root directory.
"""

import argparse
import logging
import os
import subprocess

from capbc.filestream.bucket.client import get_gpfs_bucket_mount_root


def mount_bucket(bucket_names):
    bucket2root = get_gpfs_bucket_mount_root()
    cmds = ""
    for bucket in bucket_names:
        if bucket in bucket2root:
            logging.info(
                f"{bucket} already been mounted to {bucket2root[bucket]}."
            )

        else:
            cmds += "hitc mount exec %s\n" % bucket

    proc = subprocess.Popen(
        ["/bin/sh"],
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    if cmds:
        try:
            _, errs = proc.communicate(cmds.encode("utf-8"), timeout=15)
            assert errs is None, (
                f"Got error: {errs}.\n"
                f"Please make sure your have the right to mount the bucket."
            )
        except TimeoutError:
            proc.kill()
            proc.communicate()


def create_softlink(bucket_names):
    if len(bucket_names) > 1:
        raise ValueError(
            "Only support to create soft link for HDLTAlgorithm bucket."
        )
    else:
        assert (
            len(bucket_names) == 1
        ), f"The size of bucket should be 1, but is {len(bucket_names)}"
        bucket2root = get_gpfs_bucket_mount_root()
        bucket = bucket_names[0]
        assert bucket in bucket2root, f"{bucket} not mount."

        if bucket == "HDLTAlgorithm":
            bucket_pack_data_path = os.path.join(
                bucket2root[bucket], "data/pack_data"
            )
            bucket_orig_data_path = os.path.join(
                bucket2root[bucket], "data/orig_data"
            )
            bucket_modelzoo_path = os.path.join(
                bucket2root[bucket], "models/bayes_release_models"
            )
            tmp_bucket_links = [
                "./tmp_data",
                "./tmp_orig_data",
                "./tmp_pretrained_models",
            ]

            for link in tmp_bucket_links:
                if os.path.exists(link):
                    os.remove(link)

            cmds = (
                f"ln -s {bucket_pack_data_path} ./tmp_data\n"
                f"ln -s {bucket_orig_data_path} ./tmp_orig_data\n"
                f"ln -s {bucket_modelzoo_path} ./tmp_pretrained_models\n"
                "mkdir ./tmp_models\n"
            )
        else:
            raise ValueError(
                f"Only support to create soft link for HDLTAlgorithm bucket, "
                f"but get {bucket} bucket."
            )

    proc = subprocess.Popen(
        ["/bin/sh"],
        shell=False,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        _, errs = proc.communicate(cmds.encode("utf-8"), timeout=15)
        assert errs is None, errs
        logging.info("The soft-link has been created successfully.")
    except TimeoutError:
        proc.kill()
        proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        default="HDLTAlgorithm",
        help='Bucket names, split by `,`. default "HDLTAlgorithm"',
    )
    parser.add_argument(
        "--mount",
        action="store_true",
        default=False,
        help="Weather to mount bucket",
    )
    parser.add_argument(
        "--create-link",
        action="store_true",
        default=False,
        help="Whether to create soft-link for bucket",
    )

    logging.getLogger().setLevel(logging.INFO)

    args = parser.parse_args()
    bucket_names = str(args.bucket).split(",")

    if args.mount:
        mount_bucket(bucket_names)

    if args.create_link:
        create_softlink(bucket_names)


if __name__ == "__main__":
    main()
