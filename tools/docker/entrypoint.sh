#!/bin/bash

set -e
cat <<EOF
=============================
    ChangAn Perception
=============================
EOF

# # 导入env到/etc/profile
# for item in `cat /proc/1/environ |tr '\0' '\n'`
# do
#     echo "export $item" >> /etc/profile
# done

python /workpsace/ssh_env_setup.py

service ssh start # 容器打开时打开ssh sever
nohup jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=${NOTEBOOK_APP_TOKEN} --notebook-dir=${NOTEBOOK_DIR} &

if [[ $# -eq 0 ]]; then
    exec "/bin/bash"
else
    exec "$@"
fi