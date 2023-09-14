# Copyright (c) Changan Auto. All rights reserved.
"""Parse the pruned ids distribution in the log.

Usage:

>> python tools/task_similarity.py --cfg projects/fsd/app/configs/task_simiarity.yaml

"""  # noqa
import argparse
import copy
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from loguru import logger
except ImportError as e:
    print("Please install loguru by pip install loguru.")
    raise e


bar_width = 0.25


@logger.catch
def load_config(yaml_file: str):
    config = yaml.load(open(yaml_file, "r"), Loader=yaml.SafeLoader)
    if "created_log_path" not in config:
        config["created_log_path"] = "./"
    return config


@logger.catch
def download_log_files(raw_log_files: dict):
    log_files = list(raw_log_files.values())
    if log_files is not None:
        for log_file in log_files:
            logname = os.path.basename(log_file)
            cmd_str = f"rm -rf tmp_output/raw_logs/{logname}*"
            os.system(cmd_str)
            cmd_str = f"wget -NP tmp_output/raw_logs/ {log_file}"
            os.system(cmd_str)


@logger.catch
def log_parser(file_path: str):
    """Parse the log file to get the pruned ids.

    Args:
        file_path (str): the path of the log file

    Returns:
        dict: the pruned ids in the log file
    """
    pruned_ids_dict = {}
    layer = 0
    with open(file_path) as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if "pruned ids:" in line:
                line = line.split("[")
                line = line[1].split("]")
                line = line[0].split(",")
                pruned_ids_dict[layer] = line
                layer += 1
    return pruned_ids_dict


@logger.catch
def draw_bar(
    key_lists: list,
    value_lists: list,
    labels: list,
    xlabel: str = "Layer",
    ylabel: str = "Total Number of Pruned Ids",
    saved_path: str = "./tmp.jpg",
):
    """Draw multi-cluster histogram.

    Args:
        key_lists (list): the lists of keys as x
        value_lists (list): the list of values as y
        colors (list): the color of each class
        labels (list): the label of each class
        xlabel (str, optional): the x axis label
        ylabel (str, optional): the y axis label
        saved_path (str, optional): the saved path of the figure
    """
    plt.figure()
    idx = 0
    for key, value in zip(key_lists, value_lists):
        plt.bar(
            key,
            value,
            bar_width,
            align="center",
            label=labels[idx],
            alpha=0.5,
        )
        idx += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(key_lists[0])
    plt.legend()
    plt.savefig(saved_path)


@logger.catch
def display_num_distribution(pruned_ids_dict: dict, saved_path: str):
    """Display the actual numbers of pruned ids' distribution.

    Args:
        pruned_ids_list (dct): the pruned ids list of all tasks
        saved_path (str): the saved path of the figure
    """
    nums_values = []
    keys = []
    n = 0
    copy_pruned_ids_dict = copy.deepcopy(pruned_ids_dict)
    for ids_dict in copy_pruned_ids_dict.values():
        for k, v in ids_dict.items():
            ids_dict[k] = len(v)
        nums_values.append(list(ids_dict.values()))
        keys.append([i + n * bar_width for i in list(ids_dict.keys())])
        n += 1
    draw_bar(
        keys,
        nums_values,
        labels=list(pruned_ids_dict.keys()),
        saved_path=saved_path,
    )
    logger.success("Successfully save the num distribution on {}!", saved_path)


@logger.catch
def cal_ratio(ids_list1: list, ids_list2: list):
    """Actually calculate the overlap ratio of two lists.

    Args:
        ids_list1 (list): the source list 1
        ids_list2 (list): the source list 2

    Returns:
        float: the overlap ratio of two list
    """
    intersection = list(set(ids_list1).intersection(set(ids_list2)))
    union = list(set(ids_list1).union(set(ids_list2)))
    return intersection, union, float(len(intersection) / len(union))


@logger.catch
def get_ratio_and_mean_ratio(pruned_ids_dict_1: dict, pruned_ids_dict_2: dict):
    """Get the overlap ratio and mean overlap ratio of two tasks.

    Args:
        pruned_ids_dict_1 (dict): the pruned ids of task 1
        pruned_ids_dict_2 (dict): the pruned ids of task 2

    Returns:
        list: the ratio list of two tasks' layers
    """
    ratio_list = []
    intersection_list = []
    union_list = []
    mean_ratio_1_2 = 0
    for key in pruned_ids_dict_1:
        intersection, union, ratio = cal_ratio(
            pruned_ids_dict_1[key], pruned_ids_dict_2[key]
        )
        mean_ratio_1_2 += ratio
        ratio_list.append(ratio)
        intersection_list.append(intersection)
        union_list.append(union)
    mean_ratio = mean_ratio_1_2 / len(pruned_ids_dict_2)
    logger.info("The mean ratio of ids is {}".format(mean_ratio))
    return intersection_list, union_list, ratio_list, mean_ratio


@logger.catch
def display_ratio(pruned_ids_dict: dict, saved_path: str):
    """Display the overlap ratio and mean ratio of each two tasks.

    Args:
        pruned_ids_list (list): the pruned ids list of all tasks
        saved_path (str): the saved path of the figure
    """
    all_tasks_ratio = {}
    heatmap = [
        [1 for i in range(len(pruned_ids_dict))]
        for j in range(len(pruned_ids_dict))
    ]
    for i in range(len(pruned_ids_dict)):
        for j in range(len(pruned_ids_dict)):
            if i == j:
                continue
            k = "{} and {}".format(
                list(pruned_ids_dict.keys())[i],
                list(pruned_ids_dict.keys())[j],
            )
            logger.info("========== {} ===========".format(k))
            intersection, _, ratio, mean_ratio = get_ratio_and_mean_ratio(
                list(pruned_ids_dict.values())[i],
                list(pruned_ids_dict.values())[j],
            )
            all_tasks_ratio[k] = mean_ratio
            heatmap[i][j] = mean_ratio
            heatmap[j][i] = mean_ratio
            logger.info("Intersection of each layer:")
            for idx in range(len(intersection)):
                logger.info(str(idx) + ": " + str(intersection[idx]))
            logger.info("Ratio of each layer:")
            for idx in range(len(ratio)):
                logger.info(str(idx) + ": " + str(ratio[idx]))

    logger.info("========== the matrix of all task pairs ===========")
    labels = list(pruned_ids_dict.keys())
    df = pd.DataFrame(heatmap, index=labels, columns=labels)
    logger.info("\n" + str(df))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    im = ax.imshow(df, cmap="YlGn")
    plt.colorbar(im)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(heatmap[i][j])[:4],
                ha="center",
                va="center",
                color="black",
            )
    plt.title("Task Similarity")
    plt.tight_layout()
    plt.savefig(saved_path)
    logger.success("The heatmap is saved on {}".format(saved_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="the config path",
    )

    start_arg = parser.parse_args()
    cfg = start_arg.cfg

    config = load_config(cfg)
    os.chdir(os.path.pardir)
    base_saved_path = config["base_saved_path"]
    if base_saved_path[-1] != "/":
        base_saved_path += "/"
    saved_log_path = base_saved_path + config["log"]
    f = open(saved_log_path, "w")
    logger.add(saved_log_path, format="{time} {level} {message}", level="INFO")

    download_log_files(config["raw_log"])
    log_folder_path = "tmp_output/raw_logs"
    log_folder_list = os.listdir(log_folder_path)
    log_files = []
    for file in log_folder_list:
        log_files.append(os.path.join(log_folder_path, file))

    if not log_files:
        logger.error("The folder {} is empty!".format(log_folder_path))
        quit()

    bar_width *= 3
    bar_width /= len(log_files)

    pruned_ids_dict, job_names = {}, list(config["raw_log"].keys())
    logger.info("========== log parser ==========")
    for log_file_idx in range(len(log_files)):
        pruned_ids = log_parser(log_files[log_file_idx])
        pruned_ids_dict[job_names[log_file_idx]] = pruned_ids

    if config["num_distribution"]:
        logger.info("========== plt num distribution ==========")
        display_num_distribution(
            pruned_ids_dict, base_saved_path + config["num_distribution"]
        )

    if config["task_similarity"]:
        logger.info("========== plt ids overlap ratio ==========")
        display_ratio(
            pruned_ids_dict, base_saved_path + config["task_similarity"]
        )

    f.close()
