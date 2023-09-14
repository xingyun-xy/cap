#!/usr/bin/env bash
set -x
# run ./dev/prepare_develop_dev.sh in <CAP>/

# add --user if install to local, or outside virturalenv.
pip_ext="--no-cache-dir -i http://tspdemo.changan.com.cn/nexus/repository/pypi-public  --trusted-host tspdemo.changan.com.cn"
pip3 install $1 -r requirements/develop.txt ${pip_ext}
pre-commit install
