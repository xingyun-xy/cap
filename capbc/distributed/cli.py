import os
import subprocess
import argparse
import logging
import socket
import warnings
import tempfile

from capbc.utils.deprecate import deprecated_waring

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', type=str, required=True,
                        help="Number of total workers")
    parser.add_argument('-ppn', type=str, required=False, default=None,
                        help="Number of workers per machine")
    parser.add_argument('--hostfile', type=str, required=False, default=None)
    parser.add_argument('command', nargs='+', help='command for program')
    args, unknown = parser.parse_known_args()

    if args.ppn is not None:
        deprecated_waring("The argument `-ppn` is deprecated and will be removed in version 0.8.0")  # noqa

    return args, unknown


PASS_ENVIRONMENTS = [
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PATH",
    "LD_LIBRARY_PATH",
    "WORKING_PATH",
    "RAW_JOBNAME",
    "PBS_JOBNAME",
    "JOB_ID",
    "JOB_TYPE",
    "CLUSTER",
    "DAG_ID",
    "TASK_ID",
    "RUN_ID"
]  # noqa


def url2ip(urlfile, outfile):
    with open(urlfile, 'r') as fread, open(outfile, 'w') as fwrite:  # noqa
        for oneurl in fread.readlines():
            if not oneurl.strip():
                continue
            url = str(oneurl.strip())
            try:
                ip = socket.gethostbyname(url)
                fwrite.write(str(ip) + '\n')
            except Exception as e:
                print(f'This {url} to ip failed, error msg {e}')


def _get_mpi_version():
    msg = (
        subprocess.check_output(
            ["mpirun", "--version"], stderr=subprocess.STDOUT)
        .decode("ascii")
        .strip()
    )
    if "Open MPI" in msg:
        return "OPENMPI"
    elif "mpich" in msg:
        return "MPICH"
    elif "mvapich" in msg:
        return "MVAPICH"
    else:
        raise NotImplementedError(f"Unknown mpi version {msg}")


def get_mpi_env():
    envs = dict()

    for key in PASS_ENVIRONMENTS:
        value = os.getenv(key)
        if value is not None:
            envs[key] = value

    mpi_version = _get_mpi_version()

    cmd = ""
    if mpi_version == "OPENMPI":
        for k, v in list(envs.items()):
            cmd += " -x %s=%s" % (k, str(v))
    elif mpi_version in ["MPICH", "MVAPICH"]:
        for k, v in list(envs.items()):
            cmd += " -env %s %s" % (k, str(v))
    else:
        raise NotImplementedError(f"Unknow mpi version {mpi_version}")

    return cmd


def launch_by_mpi(np, command, hostfile=None):
    cmd = f"mpirun -np {np}"
    temp_hostfile = tempfile.NamedTemporaryFile().name
    print(temp_hostfile)
    try:
        if hostfile is not None:
            url2ip(hostfile, temp_hostfile)
            cmd = cmd + f' --hostfile {temp_hostfile}'
        env_cmd = get_mpi_env()
        if env_cmd:
            cmd = cmd + env_cmd
        cmd = cmd + ' ' + command
        try:
            subprocess.check_call(cmd.split(' '), shell=False)
        except subprocess.CalledProcessError as e:
            logger.fatal(f"subprocess({cmd}) failed({e.returncode})! {e.output}")  # noqa
            raise
    finally:
        if os.path.exists(temp_hostfile):
            os.remove(temp_hostfile)


if __name__ == '__main__':
    args, unknown = parse_args()
    launch_by_mpi(
        np=args.np,
        command=' '.join(args.command) + ' ' + ' '.join(unknown),
        hostfile=args.hostfile,
    )
