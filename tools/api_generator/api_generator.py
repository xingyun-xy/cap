import argparse
import os
import re

import yaml
from easydict import EasyDict as edict
from termcolor import cprint

__all__ = [
    "api_generator",
]

title_list = ["=", "-", "^", "*"]
max_depth_title = 3


def add_title_to_write_msg(write_msg, this_title, title_msg):
    write_msg.append(this_title + "\n")
    write_msg.append("".join(title_msg * len(this_title)) + "\n")
    write_msg.append("\n")
    return write_msg


def add_to_write_api(write_api, currentmodule, miss_in_path):
    api_str1 = ".. automodule:: "
    api_str2 = "    :members:"
    api_str3 = "    :exclude-members: "
    write_api.append(api_str1 + currentmodule + "\n")
    write_api.append(api_str2 + "\n")
    if len(miss_in_path):
        write_api.append(api_str3 + (", ".join(miss_in_path)) + "\n")
    write_api.append("\n")
    return write_api


def add_to_write_msg(write_msg, currentmodule, keep_in_all):
    module_str1 = ".. py:currentmodule:: "
    module_str2 = ".. autosummary::"
    module_str3 = "    :nosignatures:"
    write_msg.append(module_str1 + currentmodule + "\n")
    write_msg.append("\n")
    write_msg.append(module_str2 + "\n")
    write_msg.append(module_str3 + "\n")
    write_msg.append("\n")
    for module in keep_in_all:
        write_msg.append("    " + module + "\n")
        write_msg.append("\n")
    return write_msg


def search_dir_files(
    path, title_pro, miss_dir, write_msg, write_api, first_flag=False
):
    all_dir_file = os.listdir(path)
    dirs = []
    files = []
    for item in all_dir_file:
        if item in miss_dir:
            continue
        if os.path.isfile(os.path.join(path, item)):
            files.append(item)
        else:
            dirs.append(item)

    path_split = path.split("/")
    cap_idx = path_split.index("cap")
    currentmodule = ".".join(path_split[cap_idx:])
    dirs_keep = []
    miss_in_path = []

    this_title = path_split[-1]
    if first_flag:
        this_title = this_title.capitalize()

    if title_pro > max_depth_title:
        title_pro = max_depth_title
        this_title = ".".join(path_split[cap_idx + max_depth_title :])
    title_msg = title_list[title_pro]

    if "__init__.py" in files:
        init_infos = open(os.path.join(path, "__init__.py"), "r").readlines()
        in_all_flag = False
        keep_in_all = []
        num_flag = 0
        for init_info in init_infos:
            if "__all__" in init_info:
                in_all_flag = True
            if in_all_flag:
                num_flag += init_info.count("[")
                num_flag -= init_info.count("]")

                init_info_strip = init_info.strip()
                init_info_strip = re.findall('["](.*?)["]', init_info_strip)
                for init_info_tmp in init_info_strip:
                    if init_info_tmp not in miss_dir:
                        if init_info_tmp not in dirs:
                            keep_in_all.append(init_info_tmp)
                        else:
                            dirs_keep.append(init_info_tmp)
                    else:
                        miss_in_path.append(init_info_tmp)

            if num_flag == 0:
                in_all_flag = False

        if len(keep_in_all) or first_flag:
            add_title_to_write_msg(write_msg, this_title, title_msg)

        if len(keep_in_all):
            currentmodule = ".".join(path_split[cap_idx:])

            if currentmodule not in write_api:
                add_to_write_api(write_api, currentmodule, miss_in_path)

            add_to_write_msg(write_msg, currentmodule, keep_in_all)

        for dir_each in dirs_keep:
            path_each = os.path.join(path, dir_each)
            write_msg, write_api = search_dir_files(
                path_each, title_pro + 1, miss_dir, write_msg, write_api
            )

    return write_msg, write_api


def api_generator(
    cap_dir, target_dir, module_name, docstring=None, ignore=None
):
    title_pro = 0
    write_msg = []
    write_api = []
    api_title = "API Reference\n"
    write_api.append(api_title)
    write_api.append("".join(title_list[1] * len(api_title)) + "\n")
    write_api.append("\n")

    miss_key = []
    if ignore is not None:
        miss_key = ignore

    title = "cap." + module_name
    write_msg.append(title + "\n")
    title_msg = title_list[title_pro]
    write_msg.append("".join(title_msg * len(title)) + "\n")
    write_msg.append("\n")

    if docstring is not None:
        write_msg.append(docstring)
        write_msg.append("\n")
        write_msg.append("\n")

    title_pro += 1
    module_path = os.path.join(cap_dir, module_name)
    write_msg, write_api = search_dir_files(
        module_path, title_pro, miss_key, write_msg, write_api, first_flag=True
    )

    write_msg = write_msg + write_api
    output_file_name = os.path.join(target_dir, module_name + ".rst")

    with open(output_file_name, "w") as f:
        for line in write_msg:
            f.write(line)
    cprint(f"[docs] create {output_file_name}", "green")


def main(args):
    file_list = edict(
        yaml.load(open(args.api_module_list, "r"), Loader=yaml.SafeLoader)
    )

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    for module_name, module_keys in file_list.items():
        docstring = None
        ignore = None
        if "docstring" in module_keys:
            docstring = module_keys["docstring"][0]
        if "ignore" in module_keys:
            ignore = module_keys["ignore"]

        cap_dir = os.path.join(args.root, "cap")
        api_generator(cap_dir, args.target_dir, module_name, docstring, ignore)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-module-list",
        type=str,
        required=True,
        help="The api module and docstring file.",
    )
    parser.add_argument(
        "--root", type=str, default="../../", help="The root dir of CAP."
    )
    parser.add_argument(
        "--target-dir", type=str, default="../../docs/source/api_reference"
    )
    args = parser.parse_args()
    main(args)
