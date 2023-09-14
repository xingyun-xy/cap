import torch

from cap.registry import OBJECT_REGISTRY

__all__ = ["IPMSegTarget"]


@OBJECT_REGISTRY.register
class IPMSegTarget(object):
    """Generate training targets for IPM-Seg task.

    Args:
        label_name: The key corresponding to the gt seg in
            label. Default: gt_seg
    """

    def __init__(self, label_name: str = "gt_seg"):
        self.label_name = label_name

    def __call__(self, label, preds):
        target = label[self.label_name]
        if target.dim() == 4:
            target = target.squeeze(dim=-1)
        # target = target.unsqueeze(dim=1)
        targets = []
        for _ in range(len(preds)):
            targets.append(target.type(torch.long))

        if len(preds) == 1:
            targets = targets[0]
            preds = preds[0]
        loss_input = {}
        loss_input["pred"] = preds
        loss_input["target"] = targets
        return [loss_input]
