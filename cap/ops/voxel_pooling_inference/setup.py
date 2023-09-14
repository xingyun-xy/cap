from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='voxel_pooling_inference_forward_wrapper',
      ext_modules=[cpp_extension.CUDAExtension('voxel_pooling_inference_ext', ['/code/cap/cap/ops/voxel_pooling_inference/src/voxel_pooling_inference_forward.cpp', '/code/cap/cap/ops/voxel_pooling_inference/src/voxel_pooling_inference_forward_cuda.cu'])],
      # include_dirs = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\include"],
      cmdclass={'build_ext': cpp_extension.BuildExtension})