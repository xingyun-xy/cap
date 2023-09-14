# Copyright (c) Changan Auto. All rights reserved.
import os

from prettytable import PrettyTable


def compare(eval_name, eval_prediction, diff_name, diff_prediction):
    """Compare real3d eval.

    Args:
        eval_prediction: The prediction being compared.
        diff_prediction: The prediction of comparison.
    """

    tb = PrettyTable()
    tb.title = eval_prediction["name"] + " compare"
    tb.field_names = eval_prediction["header"]
    eval_data = eval_prediction["data"]
    diff_data = diff_prediction["data"]
    for i, data_dict in enumerate(diff_data):
        eval_list = []
        diff_list = []
        for key, value in data_dict.items():
            if key == eval_prediction["header"][0]:
                eval_list.append(
                    os.path.join(eval_data[i][key], "+", eval_name)
                )
                diff_list.append(os.path.join(value, "+", diff_name))
            else:
                eval_list.append(str(eval_data[i][key]))
                diff = round(value - eval_data[i][key], 4)
                diff = "+" + str(diff) if diff >= 0 else str(diff)
                diff = "(" + diff + ")"
                diff_list.append(str(value) + diff)
        tb.add_row(eval_list)
        tb.add_row(diff_list)

    return str(tb)
