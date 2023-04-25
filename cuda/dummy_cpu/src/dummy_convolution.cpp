#include <torch/extension.h>

#include <cstdint>
#include <vector>
#include <iostream>

#include "Conv2dMetadata.h"
#include "UncompressedQTensor.h"

/**
 * Dummy kernel to return an output of the expected size but uninitialized values
*/
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
    std::memset(output_data, 0, output_data_len_bytes);

    UncompressedQTensor dummy_output(
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

    return dummy_output;
}

/*
    PYBIND11 is available here: https://pybind11.readthedocs.io/en/stable/installing.html#include-with-pypi
    Requires: python3-dev (on linux systems) and cmake
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2D_forward, "Convolution 2D");
}