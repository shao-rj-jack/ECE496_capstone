// Import the appropriate PyTorch headers depending on the compilation flow being used
#ifdef COMPILE_THROUGH_PYTORCH
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif

#include <cstdint>
#include <vector>
#include "cuda.h"
#include <iostream>
#include <inttypes.h>
#include "Conv2dMetadata.h"

#define INPUT_INDEX_4D(BATCH, CHANNEL, ROW, COL) \
    ((BATCH) * num_in_channels * in_height * in_width) + \
    ((CHANNEL) * in_height * in_width) + \
    ((ROW) * in_width) + \
    (COL)

#define INPUT_INDEX_3D(CHANNEL, ROW, COL) \
    ((CHANNEL) * kernel_height * in_width) + \
    ((ROW) * in_width) + \
    (COL)

#define INPUT_INDEX_2D(ROW, COL) \
    ((ROW) * in_width) + \
    (COL)

#define WEIGHT_INDEX_4D(OUT_CHANNEL, IN_CHANNEL, ROW, COL) \
    ((OUT_CHANNEL) * num_in_channels * kernel_height * kernel_width) + \
    ((IN_CHANNEL) * kernel_height * kernel_width) + \
    ((ROW) * kernel_width) + \
    (COL)

#define WEIGHT_INDEX_2D(ROW, COL) \
    ((ROW) * kernel_width) + \
    (COL)

#define OUTPUT_INDEX_4D(BATCH, CHANNEL, ROW, COL) \
    ((BATCH) * num_out_channels * out_height * out_width) + \
    ((CHANNEL) * out_height * out_width) + \
    ((ROW) * out_width) + \
    (COL)

#define OUTPUT_INDEX_2D(ROW, COL) \
    ((ROW) * out_width) + \
    (COL)

__global__ void compute_output_kernel(
    const uint8_t* input,
    const int8_t* weights,
    torch::PackedTensorAccessor32<float, 1> bias_accessor,
    uint8_t* output,
    struct Conv2d_Metadata metadata,
    bool fuse_with_relu
);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

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

    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);
    CHECK_INPUT(stride);
    CHECK_INPUT(padding);

    Conv2d_Metadata metadata = get_validated_metadata(input, weights, bias, stride, padding, output_scale, output_zero_point);

    std::vector<int64_t> output_dims = {metadata.batch_size, metadata.num_out_channels, metadata.out_height, metadata.out_width};

    int64_t output_data_len_bytes = std::accumulate(output_dims.begin(), output_dims.end(), 1, std::multiplies<int64_t>());
    uint8_t* output_data;
    cudaMalloc(&output_data, output_data_len_bytes);
    cudaCheckErrors("");

    // Copy weight_scales and weight_zero_points arrays to CUDA, so that they're accessible inside the kernel
    {
        double* weight_scales = metadata.weights_scale;
        int64_t* weight_zero_points = metadata.weights_zero_point;
        double* new_scales;
        int64_t* new_zero_points;
        cudaMalloc(&new_scales, metadata.num_out_channels * sizeof(double));
        cudaCheckErrors("");
        cudaMemcpy(new_scales, weight_scales, metadata.num_out_channels * sizeof(double), cudaMemcpyHostToDevice);
        cudaCheckErrors("");
        cudaMalloc(&new_zero_points, metadata.num_out_channels * sizeof(int64_t));
        cudaCheckErrors("");
        cudaMemcpy(new_zero_points, weight_zero_points, metadata.num_out_channels * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaCheckErrors("");
        metadata.weights_scale = new_scales;
        metadata.weights_zero_point = new_zero_points;
        delete[] weight_scales;
        delete[] weight_zero_points;
    }

    // define number of blocks and threads/block
    const int64_t num_blocks = 128;
    const int64_t num_threads = 32;
    
    // Check sizing of input and weights to confirm it fits in shared memory
    // <weights><inputs><outputs>
    const int max_shared_mem_size = 99 * 1024;
    int64_t weights_bytes_used = metadata.kernel_height*metadata.kernel_width*metadata.num_in_channels;
    // assume num_chunk_rows==1
    int64_t input_bytes_used = metadata.in_width*metadata.num_in_channels*metadata.kernel_height;
    int64_t output_bytes_used = metadata.out_width;

    const int64_t shared_mem_size_bytes = weights_bytes_used + input_bytes_used + output_bytes_used;

    std::cout << "Shared Mem Request = " << shared_mem_size_bytes << " bytes" << std::endl;
    if (shared_mem_size_bytes > max_shared_mem_size) {
        throw std::runtime_error("Not enough shared memory to run kernal");
    }

    // Add shared memory attribute as well as params for sizing
    // <<<A, B>>> where A is number of parallel kernels being run (blocks), B is number of threads per block
    cudaFuncSetAttribute(&compute_output_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_size);
    compute_output_kernel<<<num_blocks, num_threads, shared_mem_size_bytes>>>(
        reinterpret_cast<const uint8_t*>(input.data()),
        reinterpret_cast<const int8_t*>(weights.data()),
        bias.packed_accessor32<float, 1>(),
        output_data,
        metadata,
        fuse_with_relu
    );
    cudaCheckErrors("");
    cudaDeviceSynchronize();
	cudaCheckErrors("");

    UncompressedQTensor output(
        output_dims,
        torch::kPerTensorAffine,
        {output_scale},
        {output_zero_point},
        torch::kQUInt8,
        torch::kCUDA,
        output_data,
        output_data_len_bytes
    );

    // Free data used by metadata struct
    cudaFree(metadata.weights_scale);
    cudaCheckErrors("");
    cudaFree(metadata.weights_zero_point);
    cudaCheckErrors("");

    return output;
}

__device__ inline uint8_t requantize_result(float val, float scale, int64_t zero_point) {
    // Convert val to quantized domain (with the specified scale and zero point)
    int32_t val_q = __float2int_rn(val / scale) + static_cast<int32_t>(zero_point);

    // Clamp 'val' to quantized data type's MIN/MAX values
    int32_t q_min = static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
    int32_t q_max = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
    val_q = std::min(std::max(q_min, val_q), q_max);

    return static_cast<uint8_t>(val_q);
}

__global__ void compute_output_kernel(
    const uint8_t* input,
    const int8_t* weights,
    torch::PackedTensorAccessor32<float, 1> bias_accessor,
    uint8_t* output,
    struct Conv2d_Metadata metadata,
    bool fuse_with_relu
) {

    // Init shared memory as 8 bit aligned array
    extern __shared__ uint8_t shared_mem[];

    // Unpack metadata struct
    int64_t batch_size = metadata.batch_size;
    int64_t num_in_channels = metadata.num_in_channels, num_out_channels = metadata.num_out_channels;
    int64_t in_height = metadata.in_height, in_width = metadata.in_width;
    int64_t out_height = metadata.out_height, out_width = metadata.out_width;
    int64_t kernel_height = metadata.kernel_height, kernel_width = metadata.kernel_width;
    int64_t stride_x = metadata.stride_x, stride_y = metadata.stride_y;
    int64_t padding_x = metadata.padding_x, padding_y = metadata.padding_y;

    const int64_t curr_block = blockIdx.x;
    const int64_t curr_thread = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t num_threads = blockDim.x;

    const int64_t chunk_num_rows = 1;
    const int64_t num_chunks = (out_height + chunk_num_rows - 1) / chunk_num_rows;

    float input_scale = metadata.input_scale;
    int64_t input_zero_point = static_cast<int64_t>(metadata.input_zero_point);

    double* weight_scales = metadata.weights_scale;
    int64_t* weight_zero_points = metadata.weights_zero_point;

    float output_scale = metadata.output_scale;
    int64_t output_zero_point = static_cast<int64_t>(metadata.output_zero_point);

    int64_t num_inputs_rows_to_copy = kernel_height + (stride_y * (chunk_num_rows - 1));

    int shared_mem_weights_start_index = 0;
    int shared_mem_input_start_index = shared_mem_weights_start_index + kernel_height*kernel_width*num_in_channels;
    int shared_mem_output_start_index = shared_mem_input_start_index + num_inputs_rows_to_copy*in_width*num_in_channels;

    uint8_t* _input = &shared_mem[shared_mem_input_start_index];
    int8_t* _weights = (int8_t*)&shared_mem[shared_mem_weights_start_index];
    uint8_t* _output = &shared_mem[shared_mem_output_start_index];

    // Step 3: Convolution
    for (int64_t curr_batch = 0; curr_batch < batch_size; ++curr_batch) {
        // Loop over out-channels (number of filters to apply)
        for (int64_t curr_out_channel = 0; curr_out_channel < num_out_channels; ++curr_out_channel) {
            float weight_scale = static_cast<float>(weight_scales[curr_out_channel]);
            int64_t weight_zero_point = static_cast<int64_t>(__float2int_rn(weight_zero_points[curr_out_channel]));

            // Copy the weights for the current output channel into shared memory
            if (curr_thread == 0) {
                for (int64_t curr_in_channel = 0; curr_in_channel < num_in_channels; ++curr_in_channel) {
                    for (int i = 0; i < kernel_height; ++i) {
                        for (int j = 0; j < kernel_width; ++j) {
                            _weights[WEIGHT_INDEX_4D(0, curr_in_channel, i, j)] = weights[WEIGHT_INDEX_4D(curr_out_channel, curr_in_channel, i, j)];
                        }
                    }
                }
            }
            __syncthreads();

            // Loop over all assigned chunks of the input (slide kernel across input)
            for (int64_t chunk = curr_block; chunk < num_chunks; chunk += num_blocks) {
                int64_t chunk_start_row = chunk * chunk_num_rows;

                // Copy the inputs for the current output chunk, across all input channels, into shared memory
                if (curr_thread == 0) {
                    int64_t input_start_top_left_row = (0 - padding_y) + (stride_y * chunk_start_row);
                    for (int64_t curr_in_channel = 0; curr_in_channel < num_in_channels; ++curr_in_channel) {
                        for (int64_t in_row = 0; in_row < num_inputs_rows_to_copy; ++in_row) {
                            if (input_start_top_left_row + in_row < 0 || input_start_top_left_row + in_row >= in_height) {
                                continue;
                            }
                            for (int64_t in_col = 0; in_col < in_width; ++in_col) {
                                _input[INPUT_INDEX_3D(curr_in_channel, in_row, in_col)] = input[INPUT_INDEX_4D(curr_batch, curr_in_channel, input_start_top_left_row + in_row, in_col)];
                            }
                        }
                    }
                }
                __syncthreads();

                for (int64_t idx_in_chunk = curr_thread; idx_in_chunk < chunk_num_rows * out_width; idx_in_chunk += num_threads) {
                    int64_t output_row = chunk_start_row + (idx_in_chunk / out_width);
                    if (output_row >= out_height) {
                        break;
                    }
                    int64_t output_col = idx_in_chunk % out_width;

                    int64_t kernel_top_left_row = (0 - padding_y) + (stride_y * output_row);
                    int64_t kernel_top_left_col = (0 - padding_x) + (stride_x * output_col);

                    int32_t convolution_result = 0;

                    // Loop over in-channels (performs 2D convolution across all in_channels)
                    for (int64_t curr_in_channel = 0; curr_in_channel < num_in_channels; ++curr_in_channel) {
                        // Loop over kernel (perform convolution between input and kernel)
                        for (int64_t kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
                            for (int64_t kernel_col = 0; kernel_col < kernel_width; ++kernel_col) {
                                if (kernel_top_left_row + kernel_row < 0 || kernel_top_left_row + kernel_row >= in_height || kernel_top_left_col + kernel_col < 0 || kernel_top_left_col + kernel_col >= in_width) {
                                    convolution_result += 0;
                                } else {
                                    int16_t input_elem_quantized = static_cast<int16_t>(_input[INPUT_INDEX_3D(curr_in_channel, kernel_row, kernel_top_left_col + kernel_col)]);
                                    int16_t weight_elem_quantized = static_cast<int16_t>(_weights[WEIGHT_INDEX_4D(0, curr_in_channel, kernel_row, kernel_col)]);
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

                    _output[OUTPUT_INDEX_2D(output_row - chunk_start_row, output_col)] = convolution_result_quantized;
                }
                __syncthreads();
                // Copy output to global memory
                if (curr_thread == 0) {
                    for (int curr_chunk_row = 0; curr_chunk_row < chunk_num_rows; ++curr_chunk_row) {
                        for (int output_col = 0; output_col < out_width; ++output_col) {
                            output[OUTPUT_INDEX_4D(curr_batch, curr_out_channel, chunk_start_row + curr_chunk_row, output_col)] = _output[OUTPUT_INDEX_2D(curr_chunk_row, output_col)];
                        }
                    }
                }
                __syncthreads();
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
