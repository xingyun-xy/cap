import os
from collections import OrderedDict

pipeline_test = os.environ.get("CAP_PIPELINE_TEST", "0") == "1"

if pipeline_test:
    num_steps = dict(
        with_bn=100,
    )
    warmup_steps = 0
    save_interval = 5
else:
    num_steps = dict(with_bn=200000, ) # TODO xy
    warmup_steps = 1
    save_interval = 5000  # 保存间隔

base_lr = dict(with_bn=0.0002, ) #原始0.0015

interval_by = "step"

# freeze bn config
f1 = [
    "^.*backbone",  # backbone
    "^.*fpn_neck",  # fpn
    "^.*fix_channel_neck",  # fix channel fpn
]
f2 = [
    "^.*ufpn.*neck",  # ufpn in seg and 3d tasks
    "^.*_anchor_head",  # rpn heads
]
f3 = [
    "^.*_roi[^3]*head",  # roi head (without roi 3d head)
    "^(?!(.*roi|.*anchor)).*head",  # seg and 3d heads
]
f4 = [
    "^.*_roi_3d.*head",
]

freeze_bn_modules = OrderedDict(
    with_bn=[],
    freeze_bn_1=f1,
    freeze_bn_2=f2,
    freeze_bn_3=f3,
    sparse_3d_freeze_bn_1=[],
    sparse_3d_freeze_bn_2=f4,
)
