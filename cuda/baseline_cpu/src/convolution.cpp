// Import the appropriate PyTorch headers depending on the compilation flow being used
#ifdef COMPILE_THROUGH_PYTORCH
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif

#include <cstdint>
#include <vector>
#include <iostream>

#include "Conv2dMetadata.h"

#define INPUT_INDEX_4D(BATCH, CHANNEL, ROW, COL) \
    ((BATCH) * num_in_channels * in_height * in_width) + \
    ((CHANNEL) * in_height * in_width) + \
    ((ROW) * in_width) + \
    (COL)

#define WEIGHT_INDEX_4D(OUT_CHANNEL, IN_CHANNEL, ROW, COL) \
    ((OUT_CHANNEL) * num_in_channels * kernel_height * kernel_width) + \
    ((IN_CHANNEL) * kernel_height * kernel_width) + \
    ((ROW) * kernel_width) + \
    (COL)

#define OUTPUT_INDEX_4D(BATCH, CHANNEL, ROW, COL) \
    ((BATCH) * num_out_channels * out_height * out_width) + \
    ((CHANNEL) * out_height * out_width) + \
    ((ROW) * out_width) + \
    (COL)

void compute_output(
    const uint8_t* input,
    const int8_t* weights,
    const torch::Tensor& bias,
    uint8_t* output,
    struct Conv2d_Metadata metadata,
    bool fuse_with_relu
);

UncompressedQTensor conv2D_forward(
    UncompressedQTensor input,
    UncompressedQTensor weights,
    torch::Tensor bias,
    torch::Tensor stride,
    torch::Tensor padding,
    double output_scale,
    int64_t output_zero_point,
    bool fuse_with_relu
) {

    Conv2d_Metadata metadata = get_validated_metadata(input, weights, bias, stride, padding, output_scale, output_zero_point);

    std::vector<int64_t> output_dims = {metadata.batch_size, metadata.num_out_channels, metadata.out_height, metadata.out_width};

    int64_t output_data_len_bytes = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<int64_t>());
    uint8_t* output_data = new uint8_t[output_data_len_bytes];
    compute_output(
        input.data(),
        reinterpret_cast<const int8_t*>(weights.data()),
        bias,
        output_data,
        metadata,
        fuse_with_relu
    );

    UncompressedQTensor output(
        output_dims,
        torch::kPerTensorAffine,
        {output_scale},
        {output_zero_point},
        torch::kQUInt8,
        torch::kCPU,
        output_data,
        output_data_len_bytes
    );

    // Free data used by metadata struct
    delete[] metadata.weights_scale;
    delete[] metadata.weights_zero_point;

    return output;
}

inline uint8_t requantize_result(float val, float scale, int64_t zero_point) {
    // Convert val to quantized domain (with the specified scale and zero point)
    int32_t val_q = std::lrintf(val / scale) + zero_point;

    // Clamp 'val' to quantized data type's MIN/MAX values
    int32_t q_min = static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
    int32_t q_max = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
    val_q = std::min(std::max(q_min, val_q), q_max);

    return static_cast<uint8_t>(val_q);
}

void compute_output(
    const uint8_t* input,
    const int8_t* weights,
    const torch::Tensor& bias,
    uint8_t* output,
    struct Conv2d_Metadata metadata,
    bool fuse_with_relu
) {
    // Unpack metadata struct
    int64_t batch_size = metadata.batch_size;
    int64_t num_in_channels = metadata.num_in_channels, num_out_channels = metadata.num_out_channels;
    int64_t in_height = metadata.in_height, in_width = metadata.in_width;
    int64_t kernel_height = metadata.kernel_height, kernel_width = metadata.kernel_width;
    int64_t out_height = metadata.out_height, out_width = metadata.out_width;
    int64_t stride_x = metadata.stride_x, stride_y = metadata.stride_y;
    int64_t padding_x = metadata.padding_x, padding_y = metadata.padding_y;

    auto bias_accessor = bias.accessor<float, 1>();

    double input_scale = metadata.input_scale;
    int8_t input_zero_point = static_cast<int8_t>(metadata.input_zero_point);

    double* weight_scales = metadata.weights_scale;
    int64_t* weight_zero_points = metadata.weights_zero_point;

    double output_scale = metadata.output_scale;
    int8_t output_zero_point = static_cast<int8_t>(metadata.output_zero_point);

    // Loop over batches
    for (int64_t curr_batch = 0; curr_batch < batch_size; ++curr_batch) {
        // Loop over out-channels (number of filters to apply)
        for (int64_t curr_out_channel = 0; curr_out_channel < num_out_channels; ++curr_out_channel) {
            double weight_scale = weight_scales[curr_out_channel];
            int8_t weight_zero_point = static_cast<int8_t>(weight_zero_points[curr_out_channel]);
            // std::cout << curr_out_channel << ": scale=" << weight_scale << " zero_point=" << weight_zero_point << std::endl;
            // Multiplier used for requantizing the result of the convolution before placing it in the output tensor
            // float multiplier = static_cast<float>(input_scale * weight_scale / output_scale);

            int64_t output_row = 0;
            // Loop over entire input (slide kernel across input)
            for (int64_t kernel_top_left_row = 0 - padding_y; (kernel_top_left_row + kernel_height) <= in_height + padding_y; kernel_top_left_row += stride_y, output_row++) {
                int64_t output_col = 0;
                for (int64_t kernel_top_left_col = 0 - padding_x; (kernel_top_left_col + kernel_width) <= in_width + padding_x; kernel_top_left_col += stride_x, output_col++) {
                    int32_t convolution_result = 0;
                    // Loop over in-channels (performs 2D convolution across all in_channels)
                    for (int64_t curr_in_channel = 0; curr_in_channel < num_in_channels; ++curr_in_channel) {
                        // Loop over kernel (perform convolution between input and kernel)
                        for (int64_t kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
                            for (int64_t kernel_col = 0; kernel_col < kernel_width; ++kernel_col) {
                                if (kernel_top_left_row + kernel_row < 0 || kernel_top_left_row + kernel_row >= in_height || kernel_top_left_col + kernel_col < 0 || kernel_top_left_col + kernel_col >= in_width) {
                                    convolution_result += 0;
                                } else {
                                    int16_t input_elem_quantized = static_cast<int16_t>(input[INPUT_INDEX_4D(curr_batch, curr_in_channel, kernel_top_left_row + kernel_row, kernel_top_left_col + kernel_col)]);
                                    int16_t weight_elem_quantized = static_cast<int16_t>(weights[WEIGHT_INDEX_4D(curr_out_channel, curr_in_channel, kernel_row, kernel_col)]);
                                    convolution_result += (input_elem_quantized - input_zero_point) * (weight_elem_quantized - weight_zero_point);
                                }
                            }
                        }
                    }
                    float convolution_result_fp = static_cast<float>(convolution_result) * static_cast<float>((input_scale * weight_scale));
                    convolution_result_fp += bias_accessor[curr_out_channel];
                    // TODO: Only execute below statement if using fused conv2d + relu
                    if (fuse_with_relu && convolution_result_fp < 0) {
                        convolution_result_fp = 0;
                    }
                    // Requantize convolution_result based on output's quantization parameters
                    uint8_t convolution_result_quantized = requantize_result(convolution_result_fp, output_scale, output_zero_point);
                    output[OUTPUT_INDEX_4D(curr_batch, curr_out_channel, output_row, output_col)] = convolution_result_quantized;
                }
            }
        }
    }
}

// Pybind code (to be included if compiling through PyTorch)
#ifdef COMPILE_THROUGH_PYTORCH

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2D_forward, "Convolution 2D");
}

#endif
