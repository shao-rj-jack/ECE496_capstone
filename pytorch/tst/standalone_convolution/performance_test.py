import torch
import os
import subprocess
import sys
import time

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
sys.path.append(os.path.join(repo_root, "pytorch/src"))
import CustomConvolution
import tensor_wrappers

input_size = (56, 3, 128, 128) # Batch of 56: 3 channels, 128x128 image

weight_size = (1, 3, 15, 15) # 3 input channels -> 1 output channel, 15x15 kernel

bias_size = (1)

padding = (2, 2)

dtype = torch.float32

scale_and_zero_points = (0.05, 127)

weight_scales_and_zero_points = (torch.tensor([0.0223]), torch.tensor([0]))

def test(input_size, input_scale_zero_point, weights_size, weights_scale_zero_point, bias_size, padding, dtype, output_scale_zero_point, impl_to_test):
    # Generate random tensors to be fed as inputs to the convolution
    input_tensor = torch.randn(input_size, dtype=dtype)
    input_tensor_quantized = torch.quantize_per_tensor(input_tensor, *input_scale_zero_point, dtype=torch.quint8)
    weight_tensor = torch.randn(weights_size, dtype=dtype)
    weight_tensor_quantized = torch.quantize_per_channel(weight_tensor, *weights_scale_zero_point, axis=0, dtype=torch.qint8)
    bias_tensor = torch.randn(bias_size, dtype=dtype)

    output_scale, output_zero_point = output_scale_zero_point

    start_time = time.time()
    # Generate an output using the custom convolution
    # Generate an output using the custom convolution
    custom_conv_out = CustomConvolution.CustomConv2dFunction.forward(
        None,
        impl_to_test,
        tensor_wrappers.UncompressedQTensor(input_tensor_quantized),
        tensor_wrappers.UncompressedQTensor(weight_tensor_quantized),
        bias=bias_tensor,
        padding=padding,
        scale=output_scale,
        zero_point=output_zero_point
    )
    return (time.time() - start_time)

if __name__ == "__main__":
    # have 1 run first, since the first run is significantly slower, due to taking control of GPU
    print(f"Discarding first run...\n")
    test(input_size, scale_and_zero_points, weight_size, weight_scales_and_zero_points, bias_size, padding, dtype, scale_and_zero_points, "baseline_gpu")
    test_times = [0 for i in range(100)]
    for i in range(100):
        print(f"===== TEST RUN #{i + 1} =====")
        test_times[i] = test(input_size, scale_and_zero_points, weight_size, weight_scales_and_zero_points, bias_size, padding, dtype, scale_and_zero_points, "baseline_gpu")
        print(f"RUN #{i + 1} took {test_times[i]} seconds\n")
    
    total_time = 0
    for runtime in test_times:
        total_time += runtime
    
    print(f"AVERAGE TIME ACROSS ALL RUNS: {total_time / 100}\n")
