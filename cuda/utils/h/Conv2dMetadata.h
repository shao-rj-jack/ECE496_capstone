/**
 * Set of utlities that encapsulates metadata about the Conv2d operation
 * 
 * 'get_validated_metadata()' allows for parsing this metadata from the tensors
 * that are passed in to the Conv2d operation
*/

#pragma once

// Import the appropriate PyTorch headers depending on the compilation flow being used
#ifdef COMPILE_THROUGH_PYTORCH
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif

#include "UncompressedQTensor.h"

struct Conv2d_Metadata {
    int64_t batch_size;
    int64_t num_in_channels, num_out_channels;
    int64_t in_height, in_width;
    int64_t out_height, out_width;
    int64_t kernel_height, kernel_width;
    int64_t stride_x, stride_y;
    int64_t padding_x, padding_y;
    double input_scale; int64_t input_zero_point;
    double output_scale; int64_t output_zero_point;
    double* weights_scale; int64_t* weights_zero_point;
};

/**
 * Utility function that infers various metadata from the tensors passed in to the Conv2d
 * operation, and returns the values in a 'Conv2d_Metadata' struct
 * 
 * While parsing the metadata, various checks are done on the inputs to ensure that they are
 * consistent with the assumptions made while developing the various Conv2d implementations
 * 
 * Inputs: all arguments passed in to a Conv2d implementation's entry function
 * Output: filled 'Conv2d_Metadata' struct
*/
Conv2d_Metadata get_validated_metadata(
    const BaseQTensor& input,
    const BaseQTensor& weights,
    torch::Tensor bias,
    torch::Tensor stride,
    torch::Tensor padding,
    double output_scale,
    int64_t output_zero_point
);
