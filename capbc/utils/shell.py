import subprocess


__all__ = ['get_bin_path', ]


def get_bin_path(name):
    """
    Get binary file path.

    Parameters
    ----------
    name : str
        Binarary file name

    Returns
    -------
    flag : bool
        Whether the binary exists or not
    message : str
        If exists, return the binary file path. Otherwise, return the
        error message.
    """
    assert isinstance(name, str)
    p = subprocess.Popen(
        ['which', name], close_fds=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    p.wait()
    if p.returncode == 0:
        flag = True
        message = p.stdout.read().decode().strip()
    else:
        flag = False
        message = p.stderr.read().decode()
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()

    return flag, message
