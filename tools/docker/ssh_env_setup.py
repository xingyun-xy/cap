import os

all_envs = os.environ

CAP_ENV_SET = False
with open(os.path.expanduser("~/.bashrc"), "r") as fr:
    lines = [x.rstrip() for x in fr.readlines()]
    settings = list(filter(lambda x: f"CAP_ENV_SET='1'" in x, lines))
    CAP_ENV_SET = len(settings) > 0

if CAP_ENV_SET is False:
    with open(os.path.expanduser("~/.bashrc"), "a") as f:
        for k, v in all_envs.items():
            f.write('export {0}="{1}"\n'.format(k, v))
        f.write(f"export CAP_ENV_SET='1'\n")
        f.write(
            f"export LD_LIBRARY_PATH=/usr/local/TensorRT-8.4.3/lib:$LD_LIBRARY_PATH\n"
        )
