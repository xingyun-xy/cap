import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
root = (
    # "../data/train"
    "/data/train" if is_local_train else "/bucket/datasets")
root_bev = (
    # "../data/train"
    "../data/changan_data" if is_local_train else "/bucket/datasets")
# ----- Required -----

datapaths = dict(
    bev_depth=dict(
        dict(
            train_data_paths=[
                dict(
                   # 25040 samples
                    img_path=f"/tmp/dataset/dataset-3363-version-1",                        # noqa 云平台训练
                    anno_path=f"/tmp/dataset/dataset-3363-version-1/annotation",       # noqa 云平台训练
                    sample_weight=1,
                ),
                # dict(
                #     # 7980 samples
                #     img_path=f"/tmp/dataset/dataset-3480-version-1",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3480-version-1/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                # dict(
                #    # 8000 samples
                #     img_path=f"/tmp/dataset/dataset-3537-version-1",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3537-version-1/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                dict(
                   # 10980 samples
                    img_path=f"/tmp/dataset/dataset-3551-version-1",                        # noqa 云平台训练
                    anno_path=f"/tmp/dataset/dataset-3551-version-1/annotation",       # noqa 云平台训练
                    sample_weight=1,
                ),
                # dict(
                #    # 10520 samples
                #     img_path=f"/tmp/dataset/dataset-3557-version-1",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3557-version-1/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                # dict(
                #    # 10880 samples
                #     img_path=f"/tmp/dataset/dataset-3557-version-2",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3557-version-2/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                # dict(
                #    # 13700 samples
                #     img_path=f"/tmp/dataset/dataset-3557-version-3",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3557-version-3/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                # dict(
                #    # 5740 samples
                #     img_path=f"/tmp/dataset/dataset-3557-version-4",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3557-version-4/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                # dict(
                #    # 5640 samples
                #     img_path=f"/tmp/dataset/dataset-3698-version-1",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3698-version-1/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                # dict(
                #    # 5420 samples
                #     img_path=f"/tmp/dataset/dataset-3557-version-5",                        # noqa 云平台训练
                #     anno_path=f"/tmp/dataset/dataset-3557-version-5/annotation",       # noqa 云平台训练
                #     sample_weight=1,
                # ),
                dict(
                   # 2420 samples
                    img_path=f"/tmp/dataset/dataset-3804-version-1",                        # noqa 云平台训练
                    anno_path=f"/tmp/dataset/dataset-3804-version-1/annotation",       # noqa 云平台训练
                    sample_weight=1,
                ),
                dict(
                   # 7880 samples
                    img_path=f"/tmp/dataset/dataset-3805-version-1",                        # noqa 云平台训练
                    anno_path=f"/tmp/dataset/dataset-3805-version-1/annotation",       # noqa 云平台训练
                    sample_weight=1,
                ),
                dict(
                   # 12460 samples
                    img_path=f"/tmp/dataset/dataset-3812-version-1",                        # noqa 云平台训练
                    anno_path=f"/tmp/dataset/dataset-3812-version-1/annotation",       # noqa 云平台训练
                    sample_weight=1,
                ),
                dict(
                   # 8980 samples
                    img_path=f"/tmp/dataset/dataset-3810-version-1",                        # noqa 云平台训练
                    anno_path=f"/tmp/dataset/dataset-3810-version-1/annotation",       # noqa 云平台训练
                    sample_weight=1,
                ),

            ],
        ),
    ),
)

datapaths = EasyDict(datapaths)
buckets = [
    "matrix",
]
