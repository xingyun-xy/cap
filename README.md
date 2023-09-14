# ChangAN Perception
1. CAP Panorama
2. CAP Vista


## Introduction
CAP aims at developing efficient and user-friendly AI(Deep Learning) algorithm toolkit(based on Pytorch APIs) for Nvidia GPU.

It also provides implementations of the state-of-the-art (SOTA) deep learning models including Image Classification, Objdect Detection, Semantic Segmentation tasks.


## Features
1. Based on the public pytorch 1.10.2
2. SOTA results reported in research papers. And all examples are compatible with Nvidia GPU.


## Installation
1. 首先请参考[Installation Guide](tools/docker/README.md)完成CAP基础开发环境的配置
2. SVN下载[_C.so](http://iov.changan.com.cn/svn/SDA/16_C2/04%20%E9%A9%BE%E9%A9%B6%E5%BA%94%E7%94%A8%E8%BD%AF%E4%BB%B6/09%20%E8%A7%86%E8%A7%89%E6%84%9F%E7%9F%A5/14%20%E5%89%8D%E8%A7%86%E5%88%86%E5%89%B2+%E5%91%A8%E8%A7%86%E5%8A%A8%E6%80%81%E7%9B%AE%E6%A0%87%E6%A8%A1%E5%9E%8B/02%20%E5%85%B1%E4%BA%AB%E6%96%87%E6%A1%A3/%E7%8E%8B%E5%AD%A6%E6%96%B9/CAP_torch1.10.2
)放到changan_plugin_pytorch目录下
3. bashrc里写入
   source /opt/conda/bin/activate base \
   export PYTHONPATH=`CAP_PATH`:${PYTHONPATH}


## CAP目录结构

| 目录                   | 说明              |
| ---------------------- | ----------------- |
| cap                    | cap框架核心代码   |
| capbc                  | 框架底层相关      |
| projects/panorama      | 周视与BEV工程代码 |
| tools                  | 相关工具          |
| changan_plugin_pytorch | pytorch自定义算子 |
| plugins                |                   |

## How to contribute code
1. Clone this repo and checkout own branch
2. Modify code and commit

Before the first git commit command, please install the develop environment by `./dev/prepare_develop_env.sh`.

After installed, `pre-commit` checks the style of code in every commit. We will see something like the following when you run `git commit`:
```
Check added large files..................................................Passed
Fix end-of-file..........................................................Passed
Trailing whitespace......................................................Passed
Check merge conflict.....................................................Passed
Check python imports.....................................................Passed
Auto format python code..................................................Passed
Check pep8...............................................................Passed
```

3. Merge request:
Developers should develop on develop branch.

4. Merge by reviewer
**Note: We welcome contributions in any form, not just code!**

5. [CAP_Docs](docs/build/html/index.html)


