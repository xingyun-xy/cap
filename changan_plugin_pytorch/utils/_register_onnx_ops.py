_onnx_opset_version = 11


def _register_custom_op():
    from torch.onnx import register_custom_op_symbolic
    from torch.onnx.symbolic_helper import parse_args

    # scale_quanti
    @parse_args("v", "v", "v", "i", "i", "i", "b", "b", "s", "s")
    def symbolic_quantize(
        g,
        data,
        scale,
        zero_point,
        vector_dim,
        quant_min,
        quant_max,
        saturate,
        in_place,
        approximate_mode,
        march,
    ):
        return g.op(
            "::Changan/Quantize",
            data,
            scale,
            zero_point,
            vector_dim_i=vector_dim,
            quant_min_i=quant_min,
            quant_max_i=quant_max,
            saturate_i=saturate,
            in_place_i=in_place,
            approximate_mode_s=approximate_mode,
            march_str_s=march,
        )

    register_custom_op_symbolic(
        "changan::scale_quanti", symbolic_quantize, _onnx_opset_version
    )

    # scale_requanti
    @parse_args("v", "v", "v", "v", "v", "i", "s", "s", "b", "b", "s")
    def symbolic_requantize(
        g,
        input,
        in_scale,
        out_scale,
        in_zero_point,
        out_zero_point,
        vector_dim,
        input_quanti_type,
        output_quanti_type,
        pre_rshift_with_round,
        post_rshift_with_round,
        march_str,
    ):
        return g.op(
            "::Changan/Requantize",
            input,
            in_scale,
            out_scale,
            in_zero_point,
            out_zero_point,
            vector_dim_i=vector_dim,
            input_quanti_type_s=input_quanti_type,
            output_quanti_type_s=output_quanti_type,
            pre_rshift_with_round_i=pre_rshift_with_round,
            post_rshift_with_round_i=post_rshift_with_round,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::scale_requanti", symbolic_requantize, _onnx_opset_version
    )

    # quanti_resize
    @parse_args("v", "v", "v", "s", "b", "i", "i", "f", "f", "b", "s")
    def symbolic_resize(
        g,
        data,
        scale,
        zero_point,
        mode,
        align_corners,
        out_height,
        out_width,
        ratio_height,
        ratio_width,
        quantized_forward,
        march,
    ):
        return g.op(
            "::changan/Resize",
            data,
            scale,
            zero_point,
            mode_s=mode,
            align_corners_i=align_corners,
            out_height_i=out_height,
            out_width_i=out_width,
            ratio_height_f=ratio_height,
            ratio_width_f=ratio_width,
            quantized_forward_i=quantized_forward,
            march_s=march,
        )

    register_custom_op_symbolic(
        "changan::quanti_resize", symbolic_resize, _onnx_opset_version
    )

    # bpu_quanti_resize
    @parse_args("v", "s", "b", "i", "i", "f", "f", "s")
    def symbolic_bpu_resize(
        g,
        data,
        mode,
        align_corners,
        out_height,
        out_width,
        ratio_height,
        ratio_width,
        march_str,
    ):
        return g.op(
            "::changan/BpuResize",
            data,
            mode_s=mode,
            align_corners_i=align_corners,
            out_height_i=out_height,
            out_width_i=out_width,
            ratio_height_f=ratio_height,
            ratio_width_f=ratio_width,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_quanti_resize", symbolic_bpu_resize, _onnx_opset_version
    )

    # bpu_scale_quantization
    @parse_args("v", "v", "v", "i", "i", "i", "s", "s")
    def symbolic_bpu_quantize(
        g,
        data,
        scale,
        zero_point,
        vector_dim,
        quant_min,
        quant_max,
        qtype,
        march_str,
    ):
        return g.op(
            "::hoirzon/BpuQuantize",
            data,
            scale,
            zero_point,
            vector_dim_i=vector_dim,
            quant_min_i=quant_min,
            quant_max_i=quant_max,
            qtype_s=qtype,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_scale_quantization",
        symbolic_bpu_quantize,
        _onnx_opset_version,
    )

    # bpu_scale_dequantization
    @parse_args("v", "v", "i")
    def symbolic_bpu_dequantize(g, data, scale, ch_axis):
        return g.op("::changan/BpuDequantize", data, scale, ch_axis_i=ch_axis)

    register_custom_op_symbolic(
        "changan::bpu_scale_dequantization",
        symbolic_bpu_dequantize,
        _onnx_opset_version,
    )

    # bpu_scale_quanti_pooling
    @parse_args(
        "v",
        "s",
        "is",
        "is",
        "is",
        "b",
        "s",
        "s",
    )
    def symbolic_bpu_pooling(
        g,
        data,
        pool_type,
        pool_size,
        pads,
        strides,
        ceil_mode,
        out_quanti_type,
        march_str,
    ):
        return g.op(
            "::changan/BpuPool",
            data,
            pool_type_s=pool_type,
            pool_size_i=pool_size,
            pads_i=pads,
            strides_i=strides,
            ceil_mode_i=ceil_mode,
            out_quanti_type_s=out_quanti_type,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_scale_quanti_pooling",
        symbolic_bpu_pooling,
        _onnx_opset_version,
    )

    # bpu_scale_quanti_convolution
    @parse_args(
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "b",
        "i",
        "is",
        "is",
        "is",
        "is",
        "s",
        "i",
        "b",
        "b",
        "s",
        "s",
    )
    def symbolic_bpu_convolution(
        g,
        data,
        weight,
        bias,
        sumin,
        output_scale,
        accu_right_shift,
        bias_left_shift,
        output_right_shift,
        sumin_scale,
        sumin_left_shift,
        use_bias,
        filters,
        kernel_size,
        strides,
        pads,
        dilation_rate,
        activation,
        group,
        elementwise_input,
        disable_output_quantization,
        out_quanti_type,
        march_str,
    ):
        return g.op(
            "::changan/BpuConvolution",
            data,
            weight,
            bias,
            sumin,
            output_scale,
            accu_right_shift,
            bias_left_shift,
            output_right_shift,
            sumin_scale,
            sumin_left_shift,
            use_bias_i=use_bias,
            filters_i=filters,
            kernel_size_i=kernel_size,
            strides_i=strides,
            pads_i=pads,
            dilation_rate_i=dilation_rate,
            activation_s=activation,
            group_i=group,
            elementwise_input_i=elementwise_input,
            disable_output_quantization_i=disable_output_quantization,
            out_quanti_type_s=out_quanti_type,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_scale_quanti_convolution",
        symbolic_bpu_convolution,
        _onnx_opset_version,
    )

    # bgr_to_yuv444
    @parse_args("v", "b")
    def symbolic_bgr_to_yuv444(g, data, channel_reversal):
        return g.op(
            "::Changan/BgrToYuv444", data, channel_reversal_i=channel_reversal
        )

    register_custom_op_symbolic(
        "changan::bgr_to_yuv444", symbolic_bgr_to_yuv444, _onnx_opset_version
    )

    # bpu_post_process_channel_argmax
    @parse_args("v", "i", "s")
    def symbolic_bpu_post_process_channel_argmax(g, data, group, march):
        return g.op(
            "::changan/BpuPostProcessChannelArgmax",
            data,
            group_i=group,
            march_s=march,
        )

    register_custom_op_symbolic(
        "changan::bpu_post_process_channel_argmax",
        symbolic_bpu_post_process_channel_argmax,
        _onnx_opset_version,
    )

    # bpu_quanti_roi_resize
    @parse_args("v", "v", "f", "i", "i", "b", "s", "s")
    def symbolic_bpu_quanti_roi_resize(
        g,
        in_featuremap,
        in_rois,
        spatial_scale,
        out_height,
        out_width,
        aligned,
        interpolate_mode,
        march_str,
    ):
        return g.op(
            "::changan/BpuRoiResize",
            in_featuremap,
            in_rois,
            spatial_scale_f=spatial_scale,
            out_height_i=out_height,
            out_width_i=out_width,
            aligned_i=aligned,
            interpolate_mode_s=interpolate_mode,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_quanti_roi_resize",
        symbolic_bpu_quanti_roi_resize,
        _onnx_opset_version,
    )

    # quanti_roi_resize
    @parse_args("v", "v", "v", "v", "f", "i", "i", "b", "s", "b", "b", "s")
    def symbolic_quanti_roi_resize(
        g,
        featuremap,
        rois,
        scale,
        zero_point,
        spatial_scale,
        out_height,
        out_width,
        aligned,
        interpolate_mode,
        roi_quantized,
        quantized_forward,
        march_str,
    ):
        return g.op(
            "::changan/RoiResize",
            featuremap,
            rois,
            scale,
            zero_point,
            spatial_scale_f=spatial_scale,
            out_height_i=out_height,
            out_width_i=out_width,
            aligned_i=aligned,
            interpolate_mode_s=interpolate_mode,
            roi_quantized_i=roi_quantized,
            quantized_forward_i=quantized_forward,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::quanti_roi_resize",
        symbolic_quanti_roi_resize,
        _onnx_opset_version,
    )

    # bpu_scale_requantization
    @parse_args("v", "v", "v", "s", "s", "b", "s")
    def symbolic_bpu_scale_requantization(
        g,
        data,
        in_scale,
        out_scale,
        input_quanti_type,
        output_quanti_type,
        pre_rshift_with_round,
        march_str,
    ):
        return g.op(
            "::changan/BpuScaleRequantize",
            data,
            in_scale,
            out_scale,
            input_quanti_type_s=input_quanti_type,
            output_quanti_type_s=output_quanti_type,
            pre_rshift_with_round_i=pre_rshift_with_round,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_scale_requantization",
        symbolic_bpu_scale_requantization,
        _onnx_opset_version,
    )

    # bpu_scale_quanti_deconvolution
    @parse_args(
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "b",
        "i",
        "is",
        "is",
        "is",
        "is",
        "is",
        "s",
        "i",
        "b",
        "b",
        "s",
        "s",
    )
    def symbolic_bpu_scale_quanti_deconvolution(
        g,
        data,
        weight,
        bias,
        sumin,
        output_scale,
        accu_right_shift,
        bias_left_shift,
        output_right_shift,
        sumin_scale,
        sumin_left_shift,
        use_bias,
        filters,
        kernel_size,
        strides,
        pads,
        output_padding,
        dilation_rate,
        activation,
        group,
        elementwise_input,
        disable_output_quantization,
        out_quanti_type,
        march_str,
    ):
        return g.op(
            "::changan/BpuDeConvolution",
            data,
            weight,
            bias,
            sumin,
            output_scale,
            accu_right_shift,
            bias_left_shift,
            output_right_shift,
            sumin_scale,
            sumin_left_shift,
            use_bias_i=use_bias,
            filters_i=filters,
            kernel_size_i=kernel_size,
            strides_i=strides,
            pads_i=pads,
            output_padding_i=output_padding,
            dilation_rate_i=dilation_rate,
            activation_s=activation,
            group_i=group,
            elementwise_input_i=elementwise_input,
            disable_output_quantization_i=disable_output_quantization,
            out_quanti_type_s=out_quanti_type,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_scale_quanti_deconvolution",
        symbolic_bpu_scale_quanti_deconvolution,
        _onnx_opset_version,
    )

    # quanti_grid_sample
    @parse_args("v", "v", "v", "s", "s", "b", "i", "s")
    def symbolic_quanti_grid_sample(
        g,
        in_featuremap,
        in_grid,
        scale,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march_str,
    ):
        return g.op(
            "::changan/GridSample",
            in_featuremap,
            in_grid,
            scale,
            mode_s=mode,
            padding_mode_s=padding_mode,
            align_corners_i=align_corners,
            coord_shift_i=coord_shift,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::quanti_grid_sample",
        symbolic_quanti_grid_sample,
        _onnx_opset_version,
    )

    # bpu_quanti_grid_sample
    @parse_args("v", "v", "s", "s", "b", "i", "s")
    def symbolic_bpu_quanti_grid_sample(
        g,
        in_featuremap,
        in_grid,
        mode,
        padding_mode,
        align_corners,
        coord_shift,
        march_str,
    ):
        return g.op(
            "::horzion/BpuGridSample",
            in_featuremap,
            in_grid,
            mode_s=mode,
            padding_mode_s=padding_mode,
            align_corners_i=align_corners,
            coord_shift_i=coord_shift,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_quanti_grid_sample",
        symbolic_bpu_quanti_grid_sample,
        _onnx_opset_version,
    )

    # round
    @parse_args("v")
    def symbolic_round(g, data):
        return g.op("::changan.Round", data)

    register_custom_op_symbolic(
        "changan::round", symbolic_round, _onnx_opset_version
    )

    # bpu_quanti_lut
    # !!! should reorder parameters to use parse_args
    @parse_args("v", "v", "s", "i", "i", "i", "i")
    def symbolic_bpu_quanti_lut(
        g, data, table, otype, scale, bias, pshift, itplt_shift
    ):
        return g.op(
            "::changan.LUT",
            data,
            table,
            otype_s=otype,
            scale_i=scale,
            bias_i=bias,
            pshift_i=pshift,
            itplt_shift_i=itplt_shift,
        )

    register_custom_op_symbolic(
        "changan::bpu_quanti_lut", symbolic_bpu_quanti_lut, _onnx_opset_version
    )

    # bpu_quanti_line_fit
    # !!! should reorder paramters to use parse_args
    @parse_args("v", "s", "i", "i", "i")
    def symbolic_bpu_quanti_line_fit(g, data, otype, scale, bias, pshift):
        return g.op(
            "::changan.GpuLineFit",
            data,
            otype_s=otype,
            scale_i=scale,
            bias_i=bias,
            pshift_i=pshift,
        )

    register_custom_op_symbolic(
        "changan::gpu_quanti_line_fit",
        symbolic_bpu_quanti_line_fit,
        _onnx_opset_version,
    )

    # max_iou_match
    @parse_args("v", "v", "v", "f", "f", "b", "f", "b", "s")
    def symbolic_max_iou_match(
        g,
        boxes,
        gt_boxes,
        gt_boxes_num,
        pos_iou,
        neg_iou,
        allow_low_quality_match,
        low_quality_match_iou,
        legacy_bbox,
        overlap_type,
    ):
        return g.op(
            "::changan.MaxIouMatch",
            boxes,
            gt_boxes,
            gt_boxes_num,
            pos_iou_f=pos_iou,
            neg_iou_f=neg_iou,
            allow_low_quality_match_i=allow_low_quality_match,
            low_quality_match_iou_f=low_quality_match_iou,
            legacy_bbox_i=legacy_bbox,
            overlap_type_s=overlap_type,
        )

    register_custom_op_symbolic(
        "changan::max_iou_match", symbolic_max_iou_match, _onnx_opset_version
    )

    # ig_region_match
    @parse_args("v", "v", "v", "i", "f", "b", "b")
    def symbolic_ig_region_match(
        g,
        boxes,
        ig_regions,
        ig_regions_num,
        class_num,
        ig_region_overlap,
        legacy_bbox,
        output_excluding_class_id_0,
    ):
        return g.op(
            "::changan.IgRegionMatch",
            boxes,
            ig_regions,
            ig_regions_num,
            class_num_i=class_num,
            ig_region_overlap_f=ig_region_overlap,
            legacy_bbox_i=legacy_bbox,
            output_excluding_class_id_0_i=output_excluding_class_id_0,
        )

    register_custom_op_symbolic(
        "changan::ig_region_match",
        symbolic_ig_region_match,
        _onnx_opset_version,
    )

    # bpu_quanti_proposal
    @parse_args(
        "v",  # data
        "v",  # anchor
        "v",  # exp_table
        "v",  # image_sizes
        "is",  # num_anchors
        "is",  # num_classes
        "is",  # input_shifts
        "i",  # exp_shift
        "is",  # block_heights
        "is",  # block_widths
        "i",  # score_threshold
        "is",  # per_class_threshold_idxs
        "is",  # per_class_threshold_values
        "is",  # class_output_offsets
        "i",  # random_seed
        "is",  # anchor_start_offsets
        "is",  # stride_heights
        "is",  # stride_widths
        "b",  # use_clippings
        "b",  # image_size_fixed
        "i",  # image_height
        "i",  # image_width
        "s",  # im_info_type
        "i",  # nms_threshold
        "i",  # output_bbox_num
        "i",  # nms_supress_margin
        "i",  # fake_data_value
        "s",  # march_str
    )
    def symbolic_bpu_quanti_proposal(
        g,
        data,
        anchor,
        exp_table,
        image_sizes,
        num_anchors,
        num_classes,
        input_shifts,
        exp_shift,
        block_heights,
        block_widths,
        score_threshold,
        per_class_threshold_idxs,
        per_class_threshold_values,
        class_output_offsets,
        random_seed,
        anchor_start_offsets,
        stride_heights,
        stride_widths,
        use_clippings,
        image_size_fixed,
        image_height,
        image_width,
        im_info_type,
        nms_threshold,
        output_bbox_num,
        nms_supress_margin,
        fake_data_value,
        march_str,
    ):
        return g.op(
            "::changan.GpuProposal",
            data,
            anchor,
            exp_table,
            image_sizes,
            num_anchors_i=num_anchors,
            num_classes_i=num_classes,
            input_shifts_i=input_shifts,
            exp_shift_i=exp_shift,
            block_heights_i=block_heights,
            block_widths_i=block_widths,
            score_threshold_i=score_threshold,
            per_class_threshold_idxs_i=per_class_threshold_idxs,
            per_class_threshold_values_i=per_class_threshold_values,
            class_output_offsets_i=class_output_offsets,
            random_seed_i=random_seed,
            anchor_start_offsets_i=anchor_start_offsets,
            stride_heights_i=stride_heights,
            stride_widths_i=stride_widths,
            use_clippings_i=use_clippings,
            image_size_fixed_i=image_size_fixed,
            image_height_i=image_height,
            image_width_i=image_width,
            im_info_type_s=im_info_type,
            nms_threshold_i=nms_threshold,
            output_bbox_num_i=output_bbox_num,
            nms_supress_margin_i=nms_supress_margin,
            fake_data_value_i=fake_data_value,
            march_str_s=march_str,
        )

    register_custom_op_symbolic(
        "changan::bpu_quanti_proposal",
        symbolic_bpu_quanti_proposal,
        _onnx_opset_version,
    )

    # sort
    @parse_args("v", "i", "b")
    def symbolic_sort(g, data, dim, descending):
        return g.op("::changan.Sort", data, dim_i=dim, descending_i=descending)

    register_custom_op_symbolic(
        "changan::sort", symbolic_sort, _onnx_opset_version
    )

    # nms
    @parse_args("v", "v", "f")
    def symbolic_nms(g, det, scores, iou_threshold):
        return g.op(
            "::changan.Nms", det, scores, iou_threshold_f=iou_threshold
        )

    register_custom_op_symbolic(
        "changan::nms", symbolic_nms, _onnx_opset_version
    )


_register_custom_op()
