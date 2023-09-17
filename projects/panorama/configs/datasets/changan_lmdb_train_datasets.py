import os
import os.path as osp

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
root = (
    # "../data/train"
    "/data/train" if is_local_train else "/bucket/datasets")
root_bev = (
    # "../data/train"
    "../data/changan_data" if is_local_train else "/bucket/datasets")
# ----- Required -----

data_root = "/root/cap-xy/data/cabev"
datapaths = dict(
    bev_depth=dict(
        dict(
            train_data_paths=[
                # 不用 3480和3698
                # dict(
                #     img_path=osp.join(data_root, "dataset-3180-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3180-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3180-version-5"),                    
                #     anno_path=osp.join(data_root, "dataset-3180-version-5/annotation"),     
                #     sample_weight=1,
                # ),
                dict(
                    img_path=osp.join(data_root, "dataset-3218-version-1"),                    
                    anno_path=osp.join(data_root, "dataset-3218-version-1/annotation"),     
                    sample_weight=1,
                ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3237-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3237-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3363-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3363-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3377-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3377-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3537-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3537-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3551-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3551-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3557-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3557-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3557-version-2"),                    
                #     anno_path=osp.join(data_root, "dataset-3557-version-2/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3557-version-3"),                    
                #     anno_path=osp.join(data_root, "dataset-3557-version-3/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3557-version-4"),                    
                #     anno_path=osp.join(data_root, "dataset-3557-version-4/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3557-version-5"),                    
                #     anno_path=osp.join(data_root, "dataset-3557-version-5/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3804-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3804-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3805-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3805-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3810-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3810-version-1/annotation"),     
                #     sample_weight=1,
                # ),
                # dict(
                #     img_path=osp.join(data_root, "dataset-3812-version-1"),                    
                #     anno_path=osp.join(data_root, "dataset-3812-version-1/annotation"),     
                #     sample_weight=1,
                # ),
            ],
        ),
    ),
)

# data_root = "/root/cap-xy/data/cabev"
# for root, dirs, files in os.walk(data_root):


datapaths = EasyDict(datapaths)
buckets = [
    "matrix",
]
