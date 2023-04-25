#include "Conv2dMetadata.h"

Conv2d_Metadata get_validated_metadata(
    const BaseQTensor& input,
    const BaseQTensor& weights,
    torch::Tensor bias,
    torch::Tensor stride,
    torch::Tensor padding,
    double output_scale,
    int64_t output_zero_point
) {
    std::vector<int64_t> input_dims = input.sizes().vec();
    std::vector<int64_t> weights_dims = weights.sizes().vec();
    std::vector<int64_t> bias_dims = bias.sizes().vec();
    std::vector<int64_t> stride_dims = stride.sizes().vec();
    std::vector<int64_t> padding_dims = padding.sizes().vec();

    int64_t batch_size;
    int64_t num_in_channels, num_out_channels;
    int64_t in_height, in_width;
    int64_t kernel_height, kernel_width;
    int64_t stride_x, stride_y;
    int64_t padding_x, padding_y;

    // Extracting sizes from 'input' tensor
    if (input_dims.size() != 4) {
        throw std::invalid_argument("Input tensor must be rank-4 (Batch, Channels, Height, Width)");
    }
    batch_size = input_dims[0];
    num_in_channels = input_dims[1];
    in_height = input_dims[2];
    in_width = input_dims[3];
    // Extracting/verifying sizes from 'weights' tensor
    if (weights_dims.size() != 4) {
        throw std::invalid_argument("Weights tensor must be rank-4");
    }
    num_out_channels = weights_dims[0];
    if (weights_dims[1] != num_in_channels) {
        throw std::invalid_argument("Weights tensor has unexpected size for dimension 1");
    }
    kernel_height = weights_dims[2];
    kernel_width = weights_dims[3];
    // Extracting/verifying sizes from 'bias' tensor
    if (bias_dims.size() != 1) {
        throw std::invalid_argument("Bias tensor must be rank 1");
    }
    if (bias_dims[0] != num_out_channels) {
        throw std::invalid_argument("Bias tensor must be of size 'num_out_channels'");
    }
    // Misc.
    if (stride_dims.size() != 1 && stride_dims[0] == 2) {
        throw std::invalid_argument("Stride tensor must be rank-1, and length 2");
    }
    stride_y = stride[0].item<int64_t>();
    stride_x = stride[1].item<int64_t>();
    if (padding_dims.size() != 1 && padding_dims[0] == 2) {
        throw std::invalid_argument("Padding tensor must be rank-1, and length 2");
    }
    padding_y = padding[0].item<int64_t>();
    padding_x = padding[1].item<int64_t>();

    auto input_dtype = input.dtype();
    if (input_dtype != torch::kQUInt8) {
        throw std::invalid_argument("Input (activation) tensor type must be of type 'quint8'");
    }
    auto weight_dtype = weights.dtype();
    if (weight_dtype != torch::kQInt8) {
        throw std::invalid_argument("Weight tensor must be of type 'quint8'");
    }
    if (bias.dtype() != torch::kFloat32) {
        throw std::invalid_argument("Bias tensor must be of type 'float32'");
    }

    // Compute the output width and height that's compatible with the given parameters
    // (reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d,
    // with dilation and groups assumed to be 1)
    int64_t out_height = 1 + (in_height + (2 * padding_y) - (kernel_height - 1) - 1) / stride_y;
    int64_t out_width  = 1 + (in_width +  (2 * padding_x) - (kernel_width - 1)  - 1) / stride_x;

    std::vector<double> weight_qscale;
    std::vector<int64_t> weight_qzero_point;
    if (weights.qscheme() == torch::kPerChannelAffine) {
        weight_qscale = weights.q_per_channel_scales();
        weight_qzero_point = weights.q_per_channel_zero_points();
    } else {
        double scale = weights.q_scale();
        int64_t zero_point = weights.q_zero_point();
        weight_qscale = std::vector<double>(num_out_channels, scale);
        weight_qzero_point = std::vector<int64_t>(num_out_channels, zero_point);
    }
    assert(weight_qscale.size() == num_out_channels);
    assert(weight_qzero_point.size() == num_out_channels);
    double* weights_qscale_arr = new double[num_out_channels];
    std::memcpy(weights_qscale_arr, weight_qscale.data(), num_out_channels * sizeof(double));
    int64_t* weights_qzero_point_arr = new int64_t[num_out_channels];
    std::memcpy(weights_qzero_point_arr, weight_qzero_point.data(), num_out_channels * sizeof(int64_t));

    // Pack values into metadata struct
    Conv2d_Metadata metadata;
    metadata.batch_size = batch_size;
    metadata.num_in_channels = num_in_channels;
    metadata.num_out_channels = num_out_channels;
    metadata.in_height = in_height;
    metadata.in_width = in_width;
    metadata.out_height = out_height;
    metadata.out_width = out_width;
    metadata.kernel_height = kernel_height;
    metadata.kernel_width = kernel_width;
    metadata.stride_x = stride_x;
    metadata.stride_y = stride_y;
    metadata.padding_x = padding_x;
    metadata.padding_y = padding_y;
    metadata.input_scale = input.q_scale();
    metadata.weights_scale = weights_qscale_arr;
    metadata.output_scale = output_scale;
    metadata.input_zero_point = input.q_zero_point();
    metadata.weights_zero_point = weights_qzero_point_arr;
    metadata.output_zero_point = output_zero_point;

    return metadata;
}