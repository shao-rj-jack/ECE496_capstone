import torch
import numpy
import itertools
import os
import subprocess
import sys

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
sys.path.append(os.path.join(repo_root, "pytorch/src"))
import CustomConvolution
import tensor_wrappers

input_sizes = [
    (1, 1, 8, 12),       # Batch of 1: 1 channel, 8x12 image
    (1, 3, 8, 12),       # Batch of 1: 3 channels, 8x12 image
    (1, 30, 8, 12),      # Batch of 1: 30 channels, 8x12 image
    (1, 3, 128, 128),    # Batch of 1: 3 channels, 128x128 image
]

weight_sizes = [
    (1, 1, 3, 3),    # 1 input channel -> 1 output channel, 3x3 kernel
    (1, 3, 3, 5),    # 3 input channels -> 1 output channels, 3x5 kernel
    (1, 30, 3, 5),   # 30 input channels -> 1 output channels, 3x5 kernel
    (1, 3, 15, 15),  # 3 input channels -> 1 output channel, 15x15 kernel
]

bias_sizes = [
    (1),
    (1),
    (1),
    (1),
]

test_cases = zip(input_sizes, weight_sizes, bias_sizes)

padding_modes = (
    "valid",
    "same",
    1,
    (2, 2),
)

scale_and_zero_points = (
    (0.01, 0),
    (0.0254, 7),
    (0.05, 127),
)

weight_scales_and_zero_points = (
    (torch.tensor([0.03]), torch.tensor([7])),
    (torch.tensor([0.06]), torch.tensor([100])),
    (torch.tensor([0.0223]), torch.tensor([0])),
)

impls_to_test = (
    # "dummy_cpu",
    # "baseline_cpu",
    "baseline_gpu",
    "shapeshifter_gpu",
)

def test(
    input_size, input_scale_zero_point,
    weights_size, weights_scale_zero_point,
    bias_size,
    padding,
    output_scale_zero_point,
    impl_to_test,
    detailed_dumps=False,
):
    # Generate quantized random tensors to be fed as inputs to the convolution
    input_tensor = torch.randn(input_size)
    input_tensor_quantized = torch.quantize_per_tensor(input_tensor, *input_scale_zero_point, dtype=torch.quint8)
    weight_tensor = torch.randn(weights_size)
    weight_tensor_quantized = torch.quantize_per_channel(weight_tensor, *weights_scale_zero_point, axis=0, dtype=torch.qint8)
    bias_tensor = torch.randn(bias_size)

    output_scale, output_zero_point = output_scale_zero_point

    if "shapeshifter" in impl_to_test:
        group_size = 8
        assert CustomConvolution.is_shapeshifter_feasible_with_operands(input_tensor, weight_tensor, bias_tensor, 1, padding, group_size), "Operands not feasible for ShapeShifter kernel"
        shapeshifter_compression_params = tensor_wrappers.ShapeShifterCompressionParams(group_size)
        wrapped_input = tensor_wrappers.ShapeShifterCompressedQTensor(input_tensor_quantized, shapeshifter_compression_params)
        wrapped_weight = tensor_wrappers.ShapeShifterCompressedQTensor(weight_tensor_quantized, shapeshifter_compression_params)
    else:
        wrapped_input = tensor_wrappers.UncompressedQTensor(input_tensor_quantized)
        wrapped_weight = tensor_wrappers.UncompressedQTensor(weight_tensor_quantized)

    # Generate an output using the custom convolution
    custom_conv_out = CustomConvolution.CustomConv2dFunction.forward(
        None,
        impl_to_test,
        wrapped_input,
        wrapped_weight,
        bias=bias_tensor,
        padding=padding,
        scale=output_scale,
        zero_point=output_zero_point,
    )
    custom_conv_out_dequantized = torch.dequantize(custom_conv_out)
    # Generate an output using PyTorch's library implementation
    ## PyTorch 1.9 doesn't support string inputs for Conv2d's padding argument. Do the conversion here
    if type(padding) == str:
        padding = CustomConvolution.CustomConv2dFunction._parse_padding(padding, weight_tensor_quantized)
    ref_conv_out = torch.nn.quantized.functional.conv2d(input_tensor_quantized, weight_tensor_quantized, bias_tensor, padding=padding, scale=output_scale, zero_point=output_zero_point)
    ref_conv_out_dequantized = torch.dequantize(ref_conv_out)

    dump_fname_prefix = os.path.join(repo_root, "pytorch/tst/standalone_convolution/")
    dumps_to_produce = {
        "input.out"                          : input_tensor,
        "input_quantized.out"                : input_tensor_quantized,
        "input_quantized_int_repr.out"       : input_tensor_quantized.int_repr(),
        "weight.out"                         : weight_tensor,
        "weight_quantized.out"               : weight_tensor_quantized,
        "weight_quantized_int_repr.out"      : weight_tensor_quantized.int_repr(),
        "bias.out"                           : bias_tensor,
        "custom_conv_output.out"             : custom_conv_out,
        "custom_conv_output_int_repr.out"    : custom_conv_out.int_repr(),
        "custom_conv_output_dequantized.out" : custom_conv_out_dequantized,
        "ref_conv_output_int_repr.out"       : ref_conv_out.int_repr(),
        "ref_conv_output.out"                : ref_conv_out,
        "ref_conv_output_dequantized.out"    : ref_conv_out_dequantized,
    }

    if detailed_dumps:
        for base_fname, tensor in dumps_to_produce.items():
            with open(os.path.join(dump_fname_prefix, base_fname), "w") as f:
                print(tensor, file=f)

    # The dummy convolution should match the reference in all other regards except the actual values (which are all 0 for the dummy convolution)
    # To account for this, set the values of the reference to all-zero as well (all-zero in the quantized integer representation)
    if impl_to_test == "dummy_cpu":
        ref_conv_out_dequantized = (ref_conv_out_dequantized * 0) - (output_zero_point * output_scale)

    # Check for a match
    match = torch.allclose(custom_conv_out_dequantized, ref_conv_out_dequantized, atol=0.05, rtol=0.001)
    assert match, f"Difference found between custom 2D convolution and PyTorch's standard 2D convolution"


if __name__ == "__main__":
    all_test_cases = itertools.product(test_cases, scale_and_zero_points, weight_scales_and_zero_points, padding_modes, scale_and_zero_points, impls_to_test)
    for i, ((input_size, weights_size, bias_size), input_scale_zero_point, weights_scale_zero_point, padding_mode, output_scale_zero_point, impl_to_test) in enumerate(all_test_cases):
        # Print the test case:
        print(f"[{impl_to_test}]:")
        print(f"Input: {input_size},\tq_scale: {input_scale_zero_point[0]}, q_zero_point: {input_scale_zero_point[1]}")
        print(f"Weights: {weights_size},\tq_scale: {weights_scale_zero_point[0]}, q_zero_point: {weights_scale_zero_point[1]}")
        print(f"Bias: {bias_size}")
        print(f"Padding: {padding_mode}")
        print(f"Output q_scale: {output_scale_zero_point[0]}, q_zero_point: {output_scale_zero_point[1]}")
        print()
        test(
            input_size, input_scale_zero_point,
            weights_size, weights_scale_zero_point,
            bias_size,
            padding_mode,
            output_scale_zero_point,
            impl_to_test,
            detailed_dumps=False,
        )
        print(f"Test case {i} passed\n")
        # break

