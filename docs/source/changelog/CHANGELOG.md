# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [unReleased] - 2023-MM-DD

### Added

- [projects] Add model descs of auto projects.

- [data] Add docs of keys in data.

- [tools] Add main func in train.py and trainv2.py.

- [model] Add CEWithHardMining loss. 
- [config] Add face3d config.

- [model] Add proposal roi sampler. 

- [model] Add pilot task_module: `MultiStrideDepthLoss`,`ReIDClsOutputBlock`,`Camera3D`, `ReIDModule`, etc. 

- [losses] Add `L1Loss` and `MultiStrideLosses` loss, update `SmoothL1Loss`,`FocalLossV2`.

- [core] Add bev_discrete_obj to pack infer.
  
- [core] Add Panorama data structure.

- [data] Add Cocktail mmcmd dataset. 

### Changed

- [model] Update model_convert_pipeline: optimize interface design and provide more documents.

- [core] Create cap/core/affine.py. 

- [config] Change and update traj pred configs for better training.

### Deprecated
-
### Removed
-
### Fixed
-[model] Change torch.concat to torch.cat.

-[fix] Fix model training status error in validation callback.

## [1.1.0] - 2023-02-15