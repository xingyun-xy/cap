__version__ = '0.16.3+cu111.torch1102'
git_version = '370ab133f9c94ca39bc4e115a18990081b68c68e'
torch_version = '1.10.2+cu111'
torchvision_version = '0.11.3+cu111'
from .utils.version_helper import check_version
check_version('torch', torch_version, True, True)
check_version('torchvision', torchvision_version, True, True)
