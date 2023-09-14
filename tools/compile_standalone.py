#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compile pt and perf_hbm standalone, without config."""
import argparse
import multiprocessing
import os
import time

import torch
from hbdk.torch_script.tools import perf_hbm
from changan_plugin_pytorch.quantization import compile_model, export_hbir
from termcolor import cprint

try:
    from hbdk.torch_script.tools import placeholder
except ImportError:
    placeholder = None

IS_LOCAL = not os.path.exists("/running_package")


def generate_input(shape, torch_native=False):
    if torch_native:
        assert placeholder, "hbdk version should >= 3.28"
        return placeholder(shape, torch_native=True)
    else:
        return torch.randn(shape)


def get_example_input(key, input_size, torch_native=False):
    group_input_size = [
        list(map(int, current_input_size.split("x")))
        for current_input_size in input_size.split(",")
    ]
    assert len(group_input_size[0]) in [4, 5]
    if len(group_input_size) > 1:
        example_inputs = dict()  # noqa: C408
        example_inputs[key] = []
        for current_input_size in group_input_size:
            assert len(current_input_size) == 4
            example_inputs[key] += [
                generate_input(current_input_size, torch_native)
            ]
    elif len(group_input_size[0]) == 5:
        input_size = group_input_size[0]
        sequence_length = input_size[0]
        example_inputs = {
            key: [
                generate_input(input_size[1:], torch_native)
                for _ in range(sequence_length)
            ]
        }
    elif len(group_input_size[0]) == 4:
        input_size = group_input_size[0]
        example_inputs = {key: generate_input(input_size, torch_native)}
    else:
        raise NotImplementedError
    return example_inputs


def main(args):
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.pt.startswith("http://"):
        import wget

        pt_url = args.pt
        pt_file = os.path.join(out_dir, os.path.basename(pt_url))
        if os.path.exists(pt_file):
            os.system(f"rm {pt_file}")
        cprint(f"downloading to {pt_file} ...", "green")
        wget.download(pt_url, out=pt_file, bar=None)
    else:
        pt_file = args.pt

    # load pt_file
    model = torch.jit.load(pt_file)

    input_sizes = args.input_size.split(
        "^"
    )  # Separate multiple input_size with `^`
    input_keys = args.input_key.split(
        "^"
    )  # Separate multiple input_keys with `^

    torch_native = args.torch_native.split(
        "^"
    )  # Separate multiple input_keys with `^

    if len(torch_native) != 1:
        assert len(input_sizes) == len(torch_native)
    else:
        # extend the length of torch_native,equal to the length of input_sizes
        torch_native = torch_native * (len(input_sizes))

    assert len(input_sizes) == len(input_keys)
    example_inputs = {}
    for input_size, input_key, t_native in zip(
        input_sizes, input_keys, torch_native
    ):
        example_inputs.update(
            get_example_input(input_key, input_size, eval(t_native))
        )

    extra_args = args.extra_args.split()
    if extra_args:
        cprint(f"extra args are {extra_args}", "green")

    hbm = os.path.join(out_dir, f"model_opt_{args.opt}.hbm")
    hbir = os.path.join(out_dir, f"model_opt_{args.opt}.hbir")
    if IS_LOCAL:
        cpu_num = args.jobs
    else:
        cpu_num = multiprocessing.cpu_count()
    cprint(f"use {cpu_num} cpus to compile", "green")

    if os.path.exists(hbm) and args.skip_compile:
        cprint(f"{hbm} exists, skip compile", "green")
    else:
        if args.opt == "balance":
            raise NotImplementedError
        if args.opt == "O3":
            cprint(
                "You choose O3 optimize level, which may cost up "
                "to minutes, but high FPS. Refer to http://wiki.h"
                "obot.cc/pages/viewpage.action?pageId=186764640 "
                "for more details.",
                "yellow",
            )
        if export_hbir:
            cprint("exporting hbir ...", "green")
            result = export_hbir(
                module=model.eval(),
                example_inputs=example_inputs,
                hbir=hbir,
                march=args.march,
            )
        cprint("compiling model ...", "green")
        t1 = time.time()
        result = compile_model(
            module=model.eval(),
            example_inputs=example_inputs,
            march=args.march,
            input_source=[args.input_source],
            hbm=hbm,
            name=args.name,
            input_layout=args.input_layout,
            output_layout=args.output_layout,
            debug=args.debug,
            opt=args.opt,
            # # not used yet
            # balance_factor=2,
            progressbar=True,
            jobs=cpu_num,
            extra_args=extra_args,
        )
        cprint(f"compile cost {(time.time() - t1):.2f} sec", "yellow")
    cprint("perf model ...", "green")
    result = perf_hbm(  # noqa: F841
        hbm=hbm,
        layer_details=args.debug,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt", type=str, help="input pt file")
    parser.add_argument(
        "--input-size",
        type=str,
        required=True,
        help="input image size, [Nx]BxCxHxW, N is for sequence length or BxCxHxW,BxCxHxW..., separate multiple input_size with `^`",  # noqa: E501
    )
    parser.add_argument("--name", default=None, type=str, help="hbm name")
    parser.add_argument(
        "--opt",
        default="O0",
        type=str,
        help="optimize level, available options are O0, O1, O2, O3, ddr, fast, balance.",  # noqa: E501
    )  # yapf: disable
    parser.add_argument(
        "--march", default="bayes", type=str, help="march name"
    )  # noqa
    parser.add_argument(
        "--skip-compile",
        dest="skip_compile",
        action="store_true",
        help="skip compile, do perf only",
    )
    parser.add_argument(
        "--output", default="tmp_compile", type=str, help="output folder"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="compile model with debug option, show layer_details in perf result",  # noqa: E501
    )  # yapf: disable
    parser.add_argument(
        "--jobs",
        default=4,
        type=int,
        help="number of threads launched during compiler optimization."
        " Default 0 means to use all available hardware concurrency.",
    )
    parser.add_argument(
        "--input-key",
        default="img",
        type=str,
        help="dict key name in input data to model, default is `img`. Separate multiple input with `^` ",  # noqa
    )
    parser.add_argument(
        "--torch-native",
        default="False",
        type=str,
        help="Whether to use torch-native,default False.Separate multiple input_size with `^`",  # noqa
    )
    parser.add_argument(
        "--input-source",
        default="pyramid",
        type=str,
        help="input source used by hbdk, default is `pyramid`",
    )
    parser.add_argument(
        "--input-layout",
        default=None,
        choices=["NHWC", "NCHW", "BPU_RAW"],
        type=str,
        help="specify input layout of model, users can view it through `hbdk-cc -h optional`",  # noqa
    )
    parser.add_argument(
        "--output-layout",
        choices=["NHWC", "NCHW", "BPU_RAW", "ROIALIGN"],
        type=str,
        help="specify output layout of model, users can view it through `hbdk-cc -h optional`",  # noqa
    )
    parser.add_argument(
        "--extra-args",
        default="",
        type=str,
        help="extra args, like `--max-time-per-fc 1000`",
    )
    args = parser.parse_args()
    main(args)
