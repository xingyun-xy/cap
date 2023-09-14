# CAP-Docker构建
```shell
cd /path/to/dockerfile_pack_all
docker build -t cap .
```

# CAP-Docker使用说明
1. 导⼊cap镜像
2. 运⾏⾃⼰的容器
```shell
export DATA_DIR=/root/users/data_hdd/users_data
export CODE_DIR=/media/home/users/work
docker run --name xx_cap  --ipc=host --gpus all -it -v ${DATA_DIR}:/data -v ${CODE_DIR}:/code -p 8888:22 cap-trt843:v1.0
```

3. 尝试使⽤ssh登录
`ssh root@***.***.***.*** -p 8888`

4. VsCode开发环境配置
vscode远程连接服务器，需要安装 Remote-SSH，Remote Explorer插件，被连接的linux服务器上的用户主目录下面需要有.vscode-server，因网络环境，不能自动下载，需要手动配置[VsCode Server的离线安装过程](https://zhuanlan.zhihu.com/p/294933020)，远程连接配置好之后[vscode remote-ssh远程连接配置](https://blog.csdn.net/strive0_0/article/details/124967746)，进行远程连接。

vscode安装包和插件vsix文件，以及对应版本的.vscode-server文件在【10.60.1.19】服务器 **/media/home/xingyun/share/vscode** 中，可拷贝到自己文件夹下进行配置。

其他vscode相关参考：

​	[vscode免密登录](https://blog.csdn.net/qq_39683986/article/details/127221104)

​	[vscode extension market下载](https://marketplace.visualstudio.com/VSCode)
