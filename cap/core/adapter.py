# Copyright (c) Changan Auto. All rights reserved.

import torchvision

from cap.registry import OBJECT_REGISTRY

__all__ = ["TorchVisionAdapter"]


@OBJECT_REGISTRY.register
class TorchVisionAdapter(object):
    """Mapping interface of torchvision to CAP.

    Current adapter only supports transforms in torchvision.

    Args:
        interface: Func or classes in torchvision.
    """

    def __init__(self, interface, **kwargs):
        if isinstance(interface, str) and interface in dir(
            torchvision.transforms
        ):
            interface = getattr(torchvision.transforms, interface)
        assert callable(interface)
        self.interface = interface
        self.adapter = interface(**kwargs)

        if self.interface not in [
            getattr(torchvision.transforms, x)
            for x in dir(torchvision.transforms)
        ]:
            raise NotImplementedError(
                "Current adapter only support transforms in torchvision!"
            )

    def __call__(self, data):
        if self.interface in [
            getattr(torchvision.transforms, x)
            for x in dir(torchvision.transforms)
        ]:
            data["img"] = self.adapter(data["img"])
        return data
