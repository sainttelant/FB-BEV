import torch
from torch.autograd import Function
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext(
    "_ext", ["ms_deform_attn_backward", "ms_deform_attn_forward"]
)


class _MultiScaleDeformableAttnFunction(Function):
    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    ):
        return g.op(
            "trt::MultiscaleDeformableAttnPlugin_TRT",
            value,
            value_spatial_shapes,
            reference_points,
            sampling_offsets,
            attention_weights,
        )

    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index, 
        sampling_locations,
        attention_weights
    ):
        N, S, M, D = value.shape
        im2col_step = N
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(
            value, 
            value_spatial_shapes, 
            value_level_start_index, 
            sampling_locations, 
            attention_weights, 
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        N, Lq, _ = output.shape
        return output.view(N, Lq, M, D)


class _MultiScaleDeformableAttnFunction2(_MultiScaleDeformableAttnFunction):
    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    ):
        return g.op(
            "MultiScaleDeformableAttnTRT2",
            value,
            value_spatial_shapes,
            reference_points,
            sampling_offsets,
            attention_weights,
        )


_multi_scale_deformable_attn_gpu = _MultiScaleDeformableAttnFunction.apply
_multi_scale_deformable_attn_gpu2 = _MultiScaleDeformableAttnFunction2.apply


def multi_scale_deformable_attn(
    value, value_spatial_shapes, reference_points, sampling_offsets, attention_weights
):
    """Multi-scale deformable attention.

    Support TensorRT plugin MultiScaleDeformableAttnTRT: FP32 and FP16(nv_half).

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        reference_points (Tensor): The reference points.
        sampling_offsets (Tensor): The offset of sampling points,
            has shape
            (bs, num_heads, num_queries, num_levels*num_points*2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    assert value.is_cuda
    return _multi_scale_deformable_attn_gpu(
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    )


def multi_scale_deformable_attn2(
    value, value_spatial_shapes, reference_points, sampling_offsets, attention_weights
):
    """Multi-scale deformable attention.

    Support TensorRT plugin MultiScaleDeformableAttnTRT2: FP32 and FP16(nv_half2).

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        reference_points (Tensor): The reference points.
        sampling_offsets (Tensor): The offset of sampling points,
            has shape
            (bs, num_heads, num_queries, num_levels*num_points*2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points).

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    assert value.is_cuda
    return _multi_scale_deformable_attn_gpu2(
        value,
        value_spatial_shapes,
        reference_points,
        sampling_offsets,
        attention_weights,
    )
