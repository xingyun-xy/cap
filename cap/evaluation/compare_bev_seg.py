# Copyright (c) Changan Auto. All rights reserved.
from prettytable import PrettyTable


def compare(eval_name, eval_prediction, diff_name, diff_prediction):
    """Compare bev_seg eval.

    Args:
        eval_prediction: The prediction being compared.
        diff_prediction: The prediction of comparison.
    """

    tb = PrettyTable()
    tb.title = "All"
    tb.field_names = ["metrics", eval_name, diff_name]
    all_acc = []
    mean_iou = []
    mean_acc = []
    diff_all_all_acc = round(
        float(diff_prediction["all_acc"]) - float(eval_prediction["all_acc"]),
        4,
    )
    diff_all_all_acc = (
        "+%.2f%%" % (diff_all_all_acc * 100)
        if diff_all_all_acc >= 0
        else "%.2f%%" % (diff_all_all_acc * 100)
    )
    diff_all_all_acc = "(" + diff_all_all_acc + ")"
    all_acc.append("all_acc")
    all_acc.append(eval_prediction["all_acc"])
    all_acc.append(diff_prediction["all_acc"] + diff_all_all_acc)

    diff_all_mean_iou = round(
        float(diff_prediction["mean_iou"])
        - float(eval_prediction["mean_iou"]),
        4,
    )
    diff_all_mean_iou = (
        "+%.2f%%" % (diff_all_mean_iou * 100)
        if diff_all_mean_iou >= 0
        else "%.2f%%" % (diff_all_mean_iou * 100)
    )
    diff_all_mean_iou = "(" + diff_all_mean_iou + ")"
    mean_iou.append("mean_iou")
    mean_iou.append(eval_prediction["mean_iou"])
    mean_iou.append(diff_prediction["mean_iou"] + diff_all_mean_iou)

    diff_all_mean_acc = round(
        float(diff_prediction["mean_acc"])
        - float(eval_prediction["mean_acc"]),
        4,
    )
    diff_all_mean_acc = (
        "+%.2f%%" % (diff_all_mean_acc * 100)
        if diff_all_mean_acc >= 0
        else "%.2f%%" % (diff_all_mean_acc * 100)
    )
    diff_all_mean_acc = "(" + diff_all_mean_acc + ")"
    mean_acc.append("mean_acc")
    mean_acc.append(eval_prediction["mean_acc"])
    mean_acc.append(diff_prediction["mean_acc"] + diff_all_mean_acc)

    tb.add_row(all_acc)
    tb.add_row(mean_iou)
    tb.add_row(mean_acc)

    calss_tb_all = "\n"
    for i, class_dict in enumerate(eval_prediction["classes"]):
        calss_tb = PrettyTable()
        class_name = class_dict["name"]
        calss_tb.title = class_name
        calss_tb.field_names = ["metrics", eval_name, diff_name]
        acc = []
        iou = []
        diff_acc = round(
            float(diff_prediction["classes"][i]["acc"])
            - float(class_dict["acc"]),
            4,
        )
        diff_acc = (
            "+%.2f%%" % (diff_acc * 100)
            if diff_acc >= 0
            else "%.2f%%" % (diff_acc * 100)
        )
        diff_acc = "(" + diff_acc + ")"
        acc.append("acc")
        acc.append(class_dict["acc"])
        acc.append(diff_prediction["classes"][i]["acc"] + diff_acc)

        diff_iou = round(
            float(diff_prediction["classes"][i]["iou"])
            - float(class_dict["iou"]),
            4,
        )
        diff_iou = (
            "+%.2f%%" % (diff_iou * 100)
            if diff_iou >= 0
            else "%.2f%%" % (diff_iou * 100)
        )
        diff_iou = "(" + diff_iou + ")"
        iou.append("iou")
        iou.append(class_dict["iou"])
        iou.append(diff_prediction["classes"][i]["iou"] + diff_iou)

        calss_tb.add_row(acc)
        calss_tb.add_row(iou)
        calss_tb_all = calss_tb_all + str(calss_tb) + "\n"

    return str(tb) + calss_tb_all
