## 目录结构

```
bin/  # 打包依赖的可执行文件
configs/  # 各个任务的打包配置
pack.py  # 打包程序入口
README.md  # 说明
```

## 准备数据

数据需要按如下格式进行准备,input_root_dir可以在各个任务的config（train.py）中指定

```bash
input_root_dir/
    images/
        xxx.jpg
        xxx.jpg
        ...
        xxx.jpg
    anno.json
```

## 输出

运行完数据打包程序后，产出的数据放置在的output_dir也可以在config中指定，输出的数据如下包含三个lmdb dataset也就是lmdb文件夹
```bash
output_dir/
    idx/
        xxx.mdb
    img/
        xxx.mdb
    anno/
        xxx.mdb
```

## 如何运行
```
python3 pack.py --config configs/cyclist_detection/train.py 
```

## 如何读数据

调用CAP 中的 DetSeg2DAnnoDataset只需要输入idx,img,anno三个lmdb的地址即可