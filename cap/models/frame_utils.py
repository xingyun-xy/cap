# Copyright (c) Changan Auto. All rights reserved.
import torch


def pad_and_split(
    x: torch.Tensor,
    num_cached_frames: int,
    replicate_type: str = "zero",
    no_cut_seqlen: bool = False,
):
    """Pad and split the training data.

    Provide input to the lipmove model with a fixed time
    window based on the batchsize and the length of the
    training samples.

    Args:
        x: [B, C, 1, L] input array.
        num_cached_frames: window length of time dimension.
        replicate_type: use 0 for pad or edge values.
        no_cut_seqlen: whether to cut samples of fixed length.


    Returns:
        torch.Tensor: [B*L, C, 1, num_cached_frames] output array.
    """
    # x: (batch_size, channel, 1, seq_len)
    # in_shape = F.shape_array(x)
    # in_shape = [in_shape[i].asnumpy()[0] for i in range(4)]
    batch_size, num_channel, seq_len = x.shape[0], x.shape[1], x.shape[3]
    dtype = x.dtype
    device = x.device
    tmp_list = []
    assert replicate_type in ["zero", "edge"]
    for i in range(num_cached_frames):
        if i == 0:
            if replicate_type == "zero":
                txpdl = torch.zeros(
                    (batch_size, num_channel, 1, num_cached_frames - 1),
                    device=device,
                    dtype=dtype,
                )
            elif replicate_type == "edge":
                pad_edge_left = x[:, :, :, 0:1]
                txpdl = torch.cat(
                    *[pad_edge_left for j in range(num_cached_frames - 1)],
                    dim=3
                )
            tx = torch.cat((txpdl, x), dim=3)
        elif i == num_cached_frames - 1:
            if replicate_type == "zero":
                txpdr = torch.zeros(
                    (batch_size, num_channel, 1, num_cached_frames - 1),
                    device=device,
                    dtype=dtype,
                )
            elif replicate_type == "edge":
                pad_edge_right = x[:, :, :, -1:]
                txpdr = torch.cat(
                    *[pad_edge_right for j in range(num_cached_frames - 1)],
                    dim=3
                )
            tx = torch.cat((x, txpdr), dim=3)
        else:
            if replicate_type == "zero":
                txpdl = torch.zeros(
                    (batch_size, num_channel, 1, num_cached_frames - 1 - i),
                    device=device,
                    dtype=dtype,
                )
                txpdr = torch.zeros(
                    (batch_size, num_channel, 1, i), device=device, dtype=dtype
                )
            elif replicate_type == "edge":
                pad_edge_left = x[:, :, :, 0:1]
                pad_edge_right = x[:, :, :, -1:]
                txpdl = torch.cat(
                    *[pad_edge_left for j in range(num_cached_frames - 1 - i)],
                    dim=3
                )
                txpdr = torch.cat(*[pad_edge_right for j in range(i)], dim=3)
            tx = torch.cat((txpdl, x, txpdr), dim=3)
        tmp_list.append(tx)

    # x: (batch_size, channel, 1, seq_len+cached_frame-1, cached_frame)
    x = torch.stack(tmp_list, axis=4)
    # x: (batch_size, channel, 1, seq_len, cached_frame)
    if not no_cut_seqlen:
        x = x[:, :, :, :seq_len, :]
    # x: (batch_size, seq_len, channel, 1, cached_frame)
    x = x.permute((0, 3, 1, 2, 4))
    # x: (batch_size * seq_len, channel, 1, cached_frame)
    x = torch.reshape(
        x, [batch_size * seq_len, num_channel, 1, num_cached_frames]
    )
    return x


def framewise_operation(x: torch.Tensor, framewise_op_str: str):
    """Difference in time dimension.

    Args:
        x: [B, L, C, 1] input array.
        framewise_op_str: methods of doing
          operations on the time dimension.


    Returns:
        torch.Tensor: [B, L, C*2, 1] output array.
    """
    if framewise_op_str == "":
        return x
    framewise_op_list = framewise_op_str.split(",")
    x_bak = torch.clone(x)
    cat_lst = [x_bak]
    # [batch_size, seq_len, x_num_channel, 1]
    seqlen = x_bak.shape[1]
    for fop in framewise_op_list:
        if fop == "fdd":  # first degree diff
            pad = torch.zeros_like(x)[:, :1, :]
            pad_x = torch.cat((pad, x), dim=1)[:, :seqlen]
            first_degree_diff = x - pad_x
            cat_lst.append(first_degree_diff)
        elif fop == "fdm":  # first degree elementwise multiplication
            pad = torch.ones_like(x)[:, :1, :]
            pad_x = torch.cat((pad, x), dim=1)[:, :seqlen]
            first_degree_elementwise_mul = x * pad_x
            cat_lst.append(first_degree_elementwise_mul)
        elif fop == "absfdd":
            pad = torch.zeros_like(x)[:, :1, :]
            pad_x = torch.cat((pad, x), dim=1)[:, :seqlen]
            abs_first_degree_diff = torch.abs(x - pad_x)
            cat_lst.append(abs_first_degree_diff)

        elif fop == "nfdd":  # normalized first degree diff
            pad = torch.zeros_like(x)[:, :1, :]
            pad_x = torch.cat((pad, x), dim=1)[:, :seqlen]
            first_degree_diff = (x - pad_x) / 2
            cat_lst.append(first_degree_diff)
        elif fop == "":
            pass
        else:
            raise NotImplementedError
    return torch.cat((cat_lst), dim=2)
