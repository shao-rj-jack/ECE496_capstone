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
#include <cassert>
#include <cmath>
#include <bitset>
#include "Conv2dMetadata.h"
#include "ShapeShifterCompressedQTensor.h"
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define V(x) for(auto i : x) std::cout << i << ", ";; std::cout << std::endl;
#define O(x) std::cout << #x << x << std::endl
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

#define DEBUG 0

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


using ShapeShifter::ShapeShifterCompressedQTensor;

__global__ void compute_output_kernel(
    const uint64_t* inputs_compressed_data,
    const uint64_t* weights_compressed_data,
    const int GROUP_SIZE,
    const int PREFIX_LEN,
	const int DATA_SIZE, 
	const int WORD_SIZE,
    torch::PackedTensorAccessor32<float, 1> bias_accessor,
    uint64_t* output_compressed_data,
    struct Conv2d_Metadata metadata,
	bool fuse_with_relu,
	const int W_total_groups,
    const int I_total_groups,	
	uint16_t* O_total_bytes_per_block,
	const int chunk_num_rows
);

std::vector<ShapeShifterCompressedQTensor> conv2D_forward(
    const ShapeShifterCompressedQTensor& input,
    const ShapeShifterCompressedQTensor& weights,
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
	
	const int GROUP_SIZE = weights.get_compression_params().group_size; 
	const int DATA_SIZE = ShapeShifter::DATA_SIZE;
	const int PREFIX_LEN = ShapeShifter::PREFIX_LEN;
	const int WORD_SIZE = ShapeShifter::WORD_SIZE;
	const int W_total_groups = weights.get_num_groups();
	const int I_total_groups = input.get_num_groups();
	const int max_shared_mem_size = 99 * 1024;
	const int chunk_num_rows = 1;
	
	assert(GROUP_SIZE <= 60);
	assert(DATA_SIZE == 8);
	assert(PREFIX_LEN == 3);
	assert(WORD_SIZE == 64);
	assert((GROUP_SIZE*DATA_SIZE)%WORD_SIZE == 0);
    
	Conv2d_Metadata metadata = get_validated_metadata(input, weights, bias, stride, padding, output_scale, output_zero_point);
	
	assert(metadata.batch_size == 1); //easy to scale using this, for simplicity assumed 1
	
    std::vector<int64_t> output_dims_per_block = {metadata.batch_size, metadata.num_out_channels, 1, metadata.out_width};
	
	//Alocate Max
	int num_output_elems_per_block = metadata.num_out_channels*metadata.out_width * chunk_num_rows; 
	int num_groups_per_block = num_output_elems_per_block/GROUP_SIZE;
	if((num_output_elems_per_block) % GROUP_SIZE) num_groups_per_block++;
	
	int num_words_per_block = (((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)*num_groups_per_block)/WORD_SIZE);
	if((((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)*num_groups_per_block)%WORD_SIZE)) num_words_per_block++;
	
	int64_t output_compressed_len = num_words_per_block*metadata.out_height;
	
	uint64_t* output_compressed;
	cudaMalloc((void**)&output_compressed, (WORD_SIZE/8) * output_compressed_len); //8 bits per byte 
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

    
	// define number of blocks and threads/block ans shared mem_size to use
    const int64_t num_blocks = metadata.out_height; //1 block does 1 row of output so this makes sense, block id used to write output row back into global memory
    const int64_t num_threads = 32;
	
	//<weights_decomp><input_group_indicies><inputs_decomp><outputs_nocomp><group_vecs><outputs_comp>
	//Note: a max limit of the uinque group_indicies / block as follows: Each block works over at most ((kernel_height*in_width)*num_in_channels)/GROUP_SIZE number of groups
	
	int total_num_weights_bytes = (W_total_groups*GROUP_SIZE*DATA_SIZE)/8; //decompressed no meta
	int max_num_I_groups_per_2d_slice = (metadata.kernel_height*metadata.in_width)/GROUP_SIZE;
	if((metadata.kernel_height*metadata.in_width)%GROUP_SIZE) max_num_I_groups_per_2d_slice++;
	int max_num_I_group_indicies = (++max_num_I_groups_per_2d_slice)*metadata.num_in_channels;
	//NOTE the above is an estimate, due to a poor alignment of group size and input size, 
	//the above can is needed -- for example, 127x (2x9) inputs, with the in width = 9, groupsize =8
	//kernel 127x (1x1) so 1 group for 8 + 1 extra group is needed per row of the input
	//Note that due to how groups align with the rows wher the extra group is split into 3 groups
	//example out_width = 10, <G1=1elems><G2=8elems<G3=1elems>, so for this reason add another 
	//group to max_I_gourps_per_2d_slice per TODO: need to check if this is optimal heuristic
	int total_num_group_indicies_bytes = max_num_I_group_indicies*2; //16 bit per indicie
	int total_num_inputs_bytes = (max_num_I_group_indicies*GROUP_SIZE*DATA_SIZE)/8; //no meta 
	if((max_num_I_group_indicies*GROUP_SIZE*DATA_SIZE)%8) total_num_inputs_bytes++;
	
	int total_num_output_nocomp_bytes = (num_groups_per_block*GROUP_SIZE*DATA_SIZE)/8; //no metadata
	if((num_groups_per_block*GROUP_SIZE*DATA_SIZE)%8) total_num_output_nocomp_bytes++;
		
	int total_group_len_bytes = (num_groups_per_block*2); //2 bytes to store len
	int MAX_WORDS_PER_GROUP = ((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)/WORD_SIZE);
	if((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)%WORD_SIZE) MAX_WORDS_PER_GROUP++;
	
	int total_num_output_comp_bytes = (num_groups_per_block*MAX_WORDS_PER_GROUP)*(WORD_SIZE/8);
	int total_other_bytes = 1*8; //Extra for broadcasting word_indicies per group
	const int64_t shared_mem_size_bytes = total_num_weights_bytes + total_num_group_indicies_bytes + total_num_inputs_bytes + total_num_output_nocomp_bytes + total_group_len_bytes + total_num_output_comp_bytes + total_other_bytes; 
	
		
	//std::cout << "Shared Mem Request = " << shared_mem_size_bytes << " bytes" << std::endl;
	if (shared_mem_size_bytes > max_shared_mem_size) {
        throw std::runtime_error("Not enough shared memory to run kernel");
    }


	// <<<A, B, C>>> where A is number of parallel kernels being run (blocks), B is number of threads per block
	//GA10x SM supported configurations
	// 128 KB L1 + 0 KB Shared Memory
	// 120 KB L1 + 8 KB Shared Memory
	// 112 KB L1 + 16 KB Shared Memory
	// 96 KB L1 + 32 KB Shared Memory
	// 64 KB L1 + 64 KB Shared Memory
	// 28 KB L1 + 100 KB Shared Memory
	//Max is 99KB 1KB is reserved for system use
	
	// Malloc some space in global memory for the total_groups and total_words
	uint16_t* d_num_bytes_outputs;
	cudaMalloc(&d_num_bytes_outputs, num_blocks*sizeof(uint16_t));
    cudaCheckErrors("");
	
	cudaFuncSetAttribute(&compute_output_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_size);
    compute_output_kernel<<<num_blocks, num_threads, shared_mem_size_bytes>>>(
        reinterpret_cast<uint64_t*>(input.data()),
        reinterpret_cast<uint64_t*>(weights.data()),
        GROUP_SIZE,
		PREFIX_LEN,
	   	DATA_SIZE,	
		WORD_SIZE,
        bias.packed_accessor32<float, 1>(),
        output_compressed,
        metadata,
		fuse_with_relu,
		W_total_groups,
		I_total_groups,
		d_num_bytes_outputs,
		chunk_num_rows
    );
	cudaCheckErrors("");
	cudaDeviceSynchronize();
	cudaCheckErrors("");

	uint16_t* h_num_bytes_outputs = (uint16_t*) malloc(num_blocks*sizeof(uint16_t));
	cudaMemcpy(h_num_bytes_outputs, d_num_bytes_outputs, num_blocks*sizeof(uint16_t), cudaMemcpyDeviceToHost);
	cudaCheckErrors("");
	//std::cout << "fwd expected num output groups_per block " << num_groups_per_block << std::endl;
	//std::cout << "fwd Max words per block " << num_words_per_block << std::endl;
	
	//Next make out_height number of sscqts which represent a vertical slice of the output tensor
	//to reduce wasted bandwidth
	//use the num_bytes_per_block vector to determine how many bytes to read in
	//otherwise would read in max evertime which would be wasteful
	//256 bit bus, so 256 bit (bus aligned) trasfers are the best
	//at most wastes one extra bus cycle -- ok
	std::vector<ShapeShifterCompressedQTensor> output_sscqt_vector; 
	
	assert(num_blocks == metadata.out_height);
	for(int blocki = 0; blocki < num_blocks; blocki++){
		
		//std::cout << "fwd function bytes in output from blocki" << h_num_bytes_outputs[blocki] << std::endl;
		assert(h_num_bytes_outputs[blocki] <= num_words_per_block*(WORD_SIZE/8));
		//TODO: remove this copying
		//should handle copying gpu->cpu inside this fwd function, and decompression inside this fwd function
		//for optimal performance
		//should return decompressed cpu side tensors for optimal performance
		uint64_t* output_compressed_shifted;
		cudaMalloc((void**)&output_compressed_shifted, (WORD_SIZE/8) * num_words_per_block); //8 bits per byte 
		cudaCheckErrors("");
		cudaMemcpy(output_compressed_shifted, output_compressed + num_words_per_block*blocki, h_num_bytes_outputs[blocki], cudaMemcpyDeviceToDevice);
		cudaCheckErrors("");

		ShapeShifterCompressedQTensor output_blocki(
			output_dims_per_block,
			torch::kPerTensorAffine,
			{output_scale},
			{output_zero_point},
			torch::kQUInt8,
			torch::kCUDA,
			(uint8_t*) output_compressed_shifted,
			(int) (h_num_bytes_outputs[blocki]),
			num_groups_per_block,
			input.get_compression_params()
    	);
		
		output_sscqt_vector.push_back(output_blocki);
			
	}

	cudaFree(d_num_bytes_outputs);
	free(h_num_bytes_outputs);
    // Free data used by metadata struct
    cudaFree(metadata.weights_scale);
    cudaCheckErrors("");
    cudaFree(metadata.weights_zero_point);
    cudaCheckErrors("");
	cudaFree(output_compressed);
	cudaCheckErrors("");
	return output_sscqt_vector;
}

__inline__ __device__ uint8_t requantize_result(float val, float scale, int64_t zero_point) {
    // Convert val to quantized domain (with the specified scale and zero point)
    int32_t val_q = __float2int_rn(val / scale) + static_cast<int32_t>(zero_point);

    // Clamp 'val' to quantized data type's MIN/MAX values
    int32_t q_min = static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
    int32_t q_max = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());

	val_q = std::min(std::max(q_min, val_q), q_max);

    return static_cast<uint8_t>(val_q);
}

__device__ void unit_decompress_into_shared(const uint64_t* GL_compressed_values, uint64_t* SH_decompressed_values, int PR_group_index, const int GROUP_SIZE, const int PREFIX_LEN, const int WORD_SIZE, /*debug*/ int64_t curr_block, int64_t curr_thread);

__device__ void unit_compress_into_shared(uint8_t* SH_uncompressed_values, uint64_t* SH_compressed_values, uint16_t* SH_group_len, const int PR_non_padded_elemets_per_thread, const int GROUP_SIZE, const int PREFIX_LEN, const int DATA_SIZE, const int WORD_SIZE, /*debug*/ int64_t curr_block, int64_t curr_thread);

__device__ void unit_compressed_to_packed(uint64_t* SH_compressed_values, uint64_t* SH_group_len, int PR_group_index1, int PR_group_index2, int PR_group_index3, const int WORD_SIZE, const int MAX_WORDS_PER_GROUP);

__global__ void compute_output_kernel(
    const uint64_t* inputs_compressed_data,
    const uint64_t* weights_compressed_data,
    const int GROUP_SIZE,
    const int PREFIX_LEN,
	const int DATA_SIZE, 
	const int WORD_SIZE,
    torch::PackedTensorAccessor32<float, 1> bias_accessor,
    uint64_t* output_compressed_data,
    struct Conv2d_Metadata metadata,
	bool fuse_with_relu,
	const int W_total_groups,
    const int I_total_groups,	
	uint16_t* O_total_bytes_per_block,
	const int chunk_num_rows
) {
	//Step 0) Intialize Shared memory as array 64 bit aligned
	//Note that when sizing shared memory in kernel call, make sure its 64 bit aligned, setup thread/block variables and other usefull
	//sharemem organization
	// <weights_decomp><input_group_indicies><inputs_decomp><outputs_nocomp><group_vecs><outputs_comp>
	//Note: Here use static and dynamic shared memory: Stackoverflow sats that dynamic memory is allocated seperatly to static
	//<static><dynamic> I use static to broadcast the #groups in this block to all threads since only thread 0 calculates this	
	extern __shared__ uint64_t sharedmem[];
	const int64_t curr_block = blockIdx.x;
    const int64_t curr_thread = threadIdx.x;
    const int64_t num_blocks = gridDim.x;
    const int64_t num_threads = blockDim.x;
	int MAX_WORDS_PER_GROUP = ((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)/WORD_SIZE);
   	if((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)%WORD_SIZE) MAX_WORDS_PER_GROUP++;	
	
	int64_t batch_size = metadata.batch_size;
    int64_t num_in_channels = metadata.num_in_channels, num_out_channels = metadata.num_out_channels;
    int64_t in_height = metadata.in_height, in_width = metadata.in_width;
    int64_t out_height = metadata.out_height, out_width = metadata.out_width;
    int64_t kernel_height = metadata.kernel_height, kernel_width = metadata.kernel_width;
    int64_t stride_x = metadata.stride_x, stride_y = metadata.stride_y;
    int64_t padding_x = metadata.padding_x, padding_y = metadata.padding_y;
	
	//Used to pick element from input
	int64_t size_input_2d = in_height * in_width;
	int64_t size_input_1d = in_width;
	

    const int64_t num_chunks = (out_height + chunk_num_rows - 1) / chunk_num_rows;

    double input_scale = metadata.input_scale;
    int8_t input_zero_point = static_cast<int8_t>(metadata.input_zero_point);

    double* weight_scales = metadata.weights_scale;
    int64_t* weight_zero_points = metadata.weights_zero_point;

    double output_scale = metadata.output_scale;
    int8_t output_zero_point = static_cast<int8_t>(metadata.output_zero_point);

 	//TODO: During decompression the global compressed values are read from by all threads * # blocks. Lot of BW -- caching may help but may need to read it all into shared memory? Currently the idea is to use L1 cache, 28 kB of it is available, might aswell use it. Depends on space available. All access are temporal/spacial close. so caching should be efficient
	
	//Step 1) Decompress weights into Shared memory
	//Input: requires number of groups (sent with compression), GROUPSIZE*DATASIZE must be 64 bit aligned.
	//Note that if they weren't it would still be ok, instead of passing 64bit pointer, I would pass 8 bit pointer
	//and I would need more calculate for the write_index (this was just dont for ease/simplification) not neccessaryy
	//another TODO for another day
	//int W_total_groups
	int SH_weights_start_index = 1;
	for(int W_unique_group_id = curr_thread; W_unique_group_id < W_total_groups; W_unique_group_id+=num_threads){
		int SH_write_index = SH_weights_start_index + (W_unique_group_id*GROUP_SIZE*DATA_SIZE)/WORD_SIZE;
		unit_decompress_into_shared(weights_compressed_data, (uint64_t*)&(sharedmem[SH_write_index]),W_unique_group_id, GROUP_SIZE, PREFIX_LEN, WORD_SIZE, curr_block, curr_thread); 
	}
	int SH_inputs_group_indicies_start_index = SH_weights_start_index + (W_total_groups*GROUP_SIZE*DATA_SIZE)/WORD_SIZE;
	//printf("(block=%d,thread=%d): weights_start_index%d\n", (int) curr_block, (int) curr_thread, SH_weights_start_index);
	//printf("(block=%d,thread=%d): input_group_i_start_index%d\n", (int) curr_block, (int) curr_thread, SH_inputs_group_indicies_start_index);
	//Step 2) Decompress Inputs into Shared memory, only does a subset of the input per block, this is done serially to prevent multiple threads from reading in redundant info
	//Note: could use mutexts to make threads communicate the groups they have read in but that is a TODO for another time
	uint64_t PR_group_indicies_written = 0;
	if(curr_thread == 0){
		for (int64_t chunk = curr_block; chunk < num_chunks; chunk += num_blocks) { //NOTE: assumes that num_block = out_height, dont need this outer loop
			int64_t chunk_start_row = chunk * chunk_num_rows;
			//chunk_num_rows is fixed to one, carrying forward for generality
			int64_t idx_in_chunk_min = curr_thread; //=0
			int64_t idx_in_chunk_max = chunk_num_rows*out_width-1;
			int64_t output_row_min = chunk_start_row + (idx_in_chunk_min/out_width);
			int64_t output_row_max = min(chunk_start_row + (idx_in_chunk_max/out_width), out_height-1);
			int64_t output_col_min = idx_in_chunk_min % out_width;
			int64_t output_col_max = idx_in_chunk_max % out_width;
			int64_t kernel_top_left_row_min = (0 - padding_y) + (stride_y * output_row_min);
			int64_t kernel_top_left_row_max = (0 - padding_y) + (stride_y * output_row_max);
			int64_t kernel_top_left_col_min = (0 - padding_x) + (stride_x * output_col_min);
			int64_t kernel_top_left_col_max = (0 - padding_x) + (stride_x * output_col_max);
			for (int64_t curr_in_channel = 0; curr_in_channel < num_in_channels; ++curr_in_channel) {
				uint32_t min_index = (curr_in_channel)*size_input_2d + max((long)0,min(kernel_top_left_row_min, in_height-1))*size_input_1d + max((long)0, min(kernel_top_left_col_min, in_width-1));
				uint32_t max_index = (curr_in_channel)*size_input_2d + max((long)0, min(kernel_top_left_row_max+kernel_height-1, in_height-1))*size_input_1d + max((long)0, min(kernel_top_left_col_max+kernel_width-1, in_width-1));
				//These represent that min/max indicies for this channel (have to do this channel by channel b/c the one block works on multiple rows of the input across all the channels)
				//need to keep track of a list of these unique indicies
				uint16_t min_group_index = min_index/GROUP_SIZE;
				uint16_t max_group_index = max_index/GROUP_SIZE;
				
				assert(max_group_index >= min_group_index);
				assert(max_group_index < I_total_groups);
				//printf("(block=%d,thread=%d):  mingroup=%d, maxgroup=%d\n", (int) curr_block, (int) curr_thread, (int) min_group_index, (int) max_group_index);
				//printf("(block=%d,thread=%d): curr_in_channel=%d, min_index=%d, max_index=%d, mingroup=%d, maxgroup=%d max=%d max2=%d ktlc=%d\n", (int) curr_block, (int) curr_thread, (int) curr_in_channel, (int) min_index, (int) max_index, (int) min_group_index, (int) max_group_index, (int) max((long)0,min(kernel_top_left_row_max, in_height-1)), (int) max((long)0, min(kernel_top_left_col_max, in_width-1)), (int) kernel_top_left_col_max);
				//Check uniqueness before inserting
				for(; min_group_index <= max_group_index; min_group_index++){
					bool PR_flag_already_written = false;
					uint16_t* s = (uint16_t*)&sharedmem[SH_inputs_group_indicies_start_index];
					for(int i = 0; i < PR_group_indicies_written; i++){
						uint16_t PR_index_already_written = s[i];
						if(min_group_index == PR_index_already_written){
							PR_flag_already_written = true;
							break;
						}
					}
					if(!PR_flag_already_written)
						s[PR_group_indicies_written++] = min_group_index;	
				}
			}
		}
		sharedmem[0] = PR_group_indicies_written; //Write here to broadcast
		/*uint16_t* s = (uint16_t*)&sharedmem[SH_inputs_group_indicies_start_index];
		for(int i = 0; i < PR_group_indicies_written; i++){
			uint16_t PR_index_already_written = s[i];
			printf("(block=%d,thread=%d): Written index=%d\n", (int) curr_block, (int) curr_thread, PR_index_already_written);
		}*/
	}
	
	__syncthreads(); //must broadcast to all threads so that they all know how many groups have been written into the sharedmem

	PR_group_indicies_written = sharedmem[0];
	int PR_group_indicies_aligned_WORD = (PR_group_indicies_written*16)/WORD_SIZE;
	if((PR_group_indicies_written*16) % WORD_SIZE) PR_group_indicies_aligned_WORD++;
	int SH_inputs_start_index = SH_inputs_group_indicies_start_index + (PR_group_indicies_aligned_WORD);
	//printf("(block=%d,thread=%d): inputs start index=%d num groups written in this block=%d\n", (int) curr_block, (int) curr_thread, (int) SH_inputs_start_index, (int) PR_group_indicies_written);	
	//Step 2.5) Have to unique group indicies that I must read in for this block. Now to actually read them in and decompress, this can be multithreaded 
	for(int PR_group_index = curr_thread; PR_group_index < PR_group_indicies_written; PR_group_index+=num_threads){
		uint16_t* s = (uint16_t*)&sharedmem[SH_inputs_group_indicies_start_index];
		uint16_t I_unique_group_id =  s[PR_group_index];
		int SH_write_index = SH_inputs_start_index + (PR_group_index*GROUP_SIZE*DATA_SIZE)/WORD_SIZE;
		unit_decompress_into_shared(inputs_compressed_data, (uint64_t*)&(sharedmem[SH_write_index]),I_unique_group_id, GROUP_SIZE, PREFIX_LEN, WORD_SIZE, curr_block, curr_thread); 
	}
	int SH_outputs_start_index = SH_inputs_start_index + (PR_group_indicies_written*GROUP_SIZE*DATA_SIZE)/WORD_SIZE;
	//printf("(block=%d,thread=%d): outputs_uc_start_index%d\n", (int) curr_block, (int) curr_thread, SH_outputs_start_index);
	//synch all threads before doing any reads on sharedmem
	__syncthreads();
	
	/*if(curr_block == 0 && curr_thread == 0){
	for(int i = 0; i < PR_group_indicies_written; i++){
		uint8_t* s = (uint8_t*)&sharedmem[SH_inputs_start_index];
		for(int j = 0; j < GROUP_SIZE; j++){		
			int16_t input_elem_quantized = static_cast<int16_t>(static_cast<uint8_t>(s[i*GROUP_SIZE + j]));
			printf("(block=%d,thread=%d): inpu_elem=%d, i=%d, j=%d\n", (int) curr_block, (int) curr_thread, input_elem_quantized, i, j);
		}
	}
	}*/
	//Step 3) Convolution
    for (int64_t curr_batch = 0; curr_batch < batch_size; ++curr_batch) {
        // Loop over out-channels (number of filters to apply)
        for (int64_t curr_out_channel = 0; curr_out_channel < num_out_channels; ++curr_out_channel) {
            float weight_scale = static_cast<float>(weight_scales[curr_out_channel]);
            int64_t weight_zero_point = static_cast<int64_t>(__float2int_rn(weight_zero_points[curr_out_channel]));
            // Loop over all assigned chunks of the input (slide kernel across input)
            for (int64_t chunk = curr_block; chunk < num_chunks; chunk += num_blocks) {
                int64_t chunk_start_row = chunk * chunk_num_rows;
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
									//So here I can calculate the group_index and the position inside the group that needs to be read in
									//need to count to position inside the group indicies vector

									int PR_index_input = curr_in_channel*size_input_2d + (kernel_top_left_row+kernel_row)*size_input_1d + (kernel_top_left_col+kernel_col);
									uint16_t PR_group_requested = PR_index_input/GROUP_SIZE;
									int PR_requested_group_position = PR_index_input % GROUP_SIZE;
									
									
									//printf("(block=%d,thread=%d): kernel_row=%ld, kernel_col=%ld, index_input=%d, groupreq=%d, posreq=%d\n", (int) curr_block, (int) curr_thread, kernel_row, kernel_col, PR_index_input, PR_group_requested, PR_requested_group_position);
										
									uint16_t* s_indicies = (uint16_t*)&sharedmem[SH_inputs_group_indicies_start_index];
									int PR_stored_group_index = -1;
									for(int PR_group_indicies_index = 0; PR_group_indicies_index < PR_group_indicies_written; PR_group_indicies_index++){
										if(PR_group_requested == s_indicies[PR_group_indicies_index]){
											PR_stored_group_index = PR_group_indicies_index;
											break;
										}
										//printf("(block=%d,thread=%d): Requested index=%d, checking index=%d\n", (int) curr_block, (int) curr_thread, PR_group_requested, s_indicies[PR_group_indicies_index]);

									}
									//printf("(block=%d,thread=%d): POST kernel_row=%ld, kernel_col=%ld, index_input=%d, groupreq=%d, posreq=%d\n", (int) curr_block, (int) curr_thread, kernel_row, kernel_col, PR_index_input, PR_group_requested, PR_requested_group_position);

									//if(PR_stored_group_index == -1)
										//printf("(block=%d,thread=%d): Requested index=%d not in block\n", (int) curr_block, (int) curr_thread, PR_group_requested);

									assert(PR_stored_group_index != -1);
									
									uint8_t* s = (uint8_t*)&sharedmem[SH_inputs_start_index];
									
                                    int16_t input_elem_quantized = static_cast<int16_t>(s[PR_stored_group_index*GROUP_SIZE + PR_requested_group_position]);
									
									s = (uint8_t*)&sharedmem[SH_weights_start_index];
									//I have ignored batch, batch = 1 assumed
                                    int16_t weight_elem_quantized = static_cast<int16_t>(reinterpret_cast<int8_t*>(s)[WEIGHT_INDEX_4D(curr_out_channel, curr_in_channel, kernel_row, kernel_col)]);	
									//printf("(block=%d,thread=%d): input elem_quan=%d weight_elem_quan=%d , kernel_row=%ld, kernel_col=%ld, requested_group=%d, stored_group=%d, position=%d\n", (int) curr_block, (int) curr_thread,(int) input_elem_quantized, (int) weight_elem_quantized, kernel_row, kernel_col, PR_group_requested, PR_stored_group_index, PR_requested_group_position);
                                    convolution_result += (input_elem_quantized - input_zero_point) * (weight_elem_quantized - weight_zero_point);
									
									//printf("(block=%d,thread=%d): POST2 kernel_row=%ld, kernel_col=%ld, index_input=%d, groupreq=%d, posreq=%d\n", (int) curr_block, (int) curr_thread, kernel_row, kernel_col, PR_index_input, PR_group_requested, PR_requested_group_position);
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
					uint8_t* s = (uint8_t*)&sharedmem[SH_outputs_start_index];
					//This writes rows contingously (across multiple channels)
					//idx_in_chunk should be right
					//printf("(block=%d,thread=%d): conv_res_fp=%f conv_res_quan=%d idxinchunk=%ld\n", (int) curr_block, (int) curr_thread, convolution_result_fp, (int) convolution_result_quantized, curr_out_channel*out_width+idx_in_chunk);

					s[curr_out_channel*out_width+idx_in_chunk] = convolution_result_quantized;
					
					//printf("(block=%d,thread=%d): WRITEPOST: conv_res_fp=%f conv_res_quan=%d idxinchunk=%ld\n", (int) curr_block, (int) curr_thread, convolution_result_fp, (int) convolution_result_quantized, curr_out_channel*out_width+idx_in_chunk);

				}
            }
        }
    }
	int PR_size_output = chunk_num_rows * out_width * num_out_channels;
	
	int PR_num_groups_output = PR_size_output/GROUP_SIZE;
	if((PR_size_output) % GROUP_SIZE) PR_num_groups_output++;
	int PR_num_groups_output_aligned_word = (PR_num_groups_output*GROUP_SIZE*DATA_SIZE)/WORD_SIZE;
	if((PR_num_groups_output*GROUP_SIZE*DATA_SIZE)%WORD_SIZE) PR_num_groups_output_aligned_word++;
	
	int SH_group_len_start_index = SH_outputs_start_index + (PR_num_groups_output_aligned_word);
 	
	//Step 4) Create space for group_len vector
	// 16bit group length works (2 bytes each) //Note: could be 1 byte to save even space
	//note that group_word_index is no longer needed since each group is placed MAX_WORDS_PER_GROUP away
	
	int PR_group_len_aligned_word = (16*PR_num_groups_output)/WORD_SIZE;
	if((16*PR_num_groups_output)%WORD_SIZE) PR_group_len_aligned_word++;
	
	int SH_outputs_compressed_start_index = SH_group_len_start_index + (PR_group_len_aligned_word);
	__syncthreads(); //synchthreads before reading outs
	
	//Step 5) Have uncompressed outputs, now to compress them and write to shared memory
	for(int unique_ucgroup_start_index = curr_thread*GROUP_SIZE; unique_ucgroup_start_index < (PR_num_groups_output*GROUP_SIZE); unique_ucgroup_start_index+=num_threads*GROUP_SIZE){
		uint8_t* s_uncompressed = (uint8_t*)&sharedmem[SH_outputs_start_index];
		uint16_t* s_group_len = (uint16_t*)&sharedmem[SH_group_len_start_index];
		
		int SH_read_index = unique_ucgroup_start_index;
		int PR_group_index = (unique_ucgroup_start_index/GROUP_SIZE);
		//dont know how many words a group will take until after compressed
		int SH_write_index = SH_outputs_compressed_start_index + (PR_group_index)*MAX_WORDS_PER_GROUP;
		int SH_group_len_index = PR_group_index;
		int PR_non_padded_elemets_per_thread = (PR_size_output - unique_ucgroup_start_index);
		if(PR_non_padded_elemets_per_thread > GROUP_SIZE) PR_non_padded_elemets_per_thread=GROUP_SIZE;
		//need to send this b/c shared mem may not be init to 0	
		//printf("(block=%d,thread=%d): called with uc_index=%d\n", (int) curr_block, (int) curr_thread, unique_ucgroup_start_index);
		unit_compress_into_shared((uint8_t*)&(s_uncompressed[SH_read_index]), (uint64_t*)&(sharedmem[SH_write_index]), (uint16_t*)&(s_group_len[SH_group_len_index]), PR_non_padded_elemets_per_thread, GROUP_SIZE, PREFIX_LEN, DATA_SIZE, WORD_SIZE, curr_block, curr_thread);
		//printf("(block=%d,thread=%d): COMPRESSPOST called with uc_index=%d\n", (int) curr_block, (int) curr_thread, unique_ucgroup_start_index);
	}
	
	__syncthreads();
	//printf("(block=%d,thread=%d): POST COMPRESS\n", (int) curr_block, (int) curr_thread);
	
	
	//Step 6) Have compressed outputs, now to pack them inplace backinto shared memory
	//this algorithm works with sets of 2 groups, requires 3 group indices that must be maintained
	//Finish condition? ceil(log_2(#groups)) interations is the max needed use that as end condition
	//each thread tries a portion of an iteration based on its threadID
	//for algo to work, have to padd group_nums to nearest power of 2, no actual padding, just something
	//to keep track (this helps with cases where and odd number of groups need to be grouped up
	//certain calls do nothing, but are needed to carry forward the algorithm	
	int PR_num_iterations = (int) ceil(log2(PR_num_groups_output));
	int PR_num_groups_padded = ((int)1<<PR_num_iterations);
	for(int PR_iteration_num = 0; PR_iteration_num < PR_num_iterations; PR_iteration_num++){
		int PR_iteration_multiplier = ((int) 1<<PR_iteration_num);
		int PR_thread_looper=0;
		while(1){
			int PR_group_index1 = PR_iteration_multiplier*((0+PR_thread_looper)+(2*curr_thread));
			int PR_group_index2 = PR_iteration_multiplier*((1+PR_thread_looper)+(2*curr_thread));
			int PR_group_index3 = PR_iteration_multiplier*((2+PR_thread_looper)+(2*curr_thread));
			
			if(PR_group_index1 >= PR_num_groups_padded) break;
			else PR_thread_looper+=num_threads*2;
			
			//Handle Padded cases before calling
			if(PR_group_index1 >= PR_num_groups_output) continue; //padded 1, padded 2, padded 3 
			if(PR_group_index2 >= PR_num_groups_output) continue; //valid 1, padded 2, padded 3 (length group_2 = 0, so do nothing)
			if(PR_group_index3 >= PR_num_groups_output) PR_group_index3 = PR_num_groups_output; //valid 1, valid 2, padded 3 (len2 is till end)
			
			unit_compressed_to_packed((uint64_t*)&sharedmem[SH_outputs_compressed_start_index], (uint64_t*)&sharedmem[SH_group_len_start_index], PR_group_index1, PR_group_index2, PR_group_index3, WORD_SIZE, MAX_WORDS_PER_GROUP);	
		}
		__syncthreads(); //syncthreads Needed to make sure all previous sets packed before proceeding 
	}
	
	//Step 7) Write compressed outputs to global memory, need to figure out the post packing size
	int PR_num_bits_packed = 0;
	for(int PR_group_index = 0; PR_group_index < PR_num_groups_output; PR_group_index++){
		uint16_t* s = (uint16_t*)&sharedmem[SH_group_len_start_index];
		PR_num_bits_packed += s[PR_group_index];
	}
	int PR_num_words_packed = PR_num_bits_packed/WORD_SIZE;
	
	if(PR_num_bits_packed % WORD_SIZE) PR_num_words_packed++;
	//printf("(block=%d,thread=%d): numbitspacked=%d\n", (int) curr_block, (int) curr_thread, (int) PR_num_bits_packed);

	//Flood the bus and write to global memory
	//num groups_output is constant ber block, b/c each block creates a constant number of elements of the output
	int MAX_OUTPUT_WORDS_PACKED_PER_BLOCK = (((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)*PR_num_groups_output)/WORD_SIZE);
	if((((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)*PR_num_groups_output)%WORD_SIZE)) MAX_OUTPUT_WORDS_PACKED_PER_BLOCK++;
	
	int GL_outputs_compressed_start_index = curr_block*MAX_OUTPUT_WORDS_PACKED_PER_BLOCK;
	for(int PR_packed_word_index = curr_thread; PR_packed_word_index < PR_num_words_packed; PR_packed_word_index+=num_threads){
		output_compressed_data[GL_outputs_compressed_start_index+PR_packed_word_index] = sharedmem[SH_outputs_compressed_start_index+PR_packed_word_index];
		//printf("(block=%d,thread=%d): index=%d packed_output=0x%016lx\n", (int) curr_block, (int) curr_thread, PR_packed_word_index, sharedmem[SH_outputs_compressed_start_index+PR_packed_word_index]);

	}
	
	//Step 8) write out metadata per block
	if(curr_thread == 0){
		O_total_bytes_per_block[curr_block] = PR_num_words_packed*(WORD_SIZE/8);
		//printf("(block=%d,thread=%d): PR_num_words_paced=%d\n", (int) curr_block, (int) curr_thread, PR_num_words_packed);
		/*if(curr_block == 0){
			printf("weights_start_index%d\n", SH_weights_start_index);
			printf("input_group_i_start_index%d\n", SH_inputs_group_indicies_start_index);
			printf("inputs_start_index%d\n", SH_inputs_start_index);
			printf("outputs_uc_start_index%d\n", SH_outputs_start_index);
			printf("group_len_start_index%d\n", SH_group_len_start_index);
			printf("outputs_c_start_index%d\n", SH_outputs_compressed_start_index);
		}*/
	}
}

//This function is to be called by each thread, it handles one group
//Input: a pointer to the compressed values, pointer to position is shared memory to write to, and the group to decompress
//Output: Write to shared memory the decompressed group, no return
__device__ void unit_decompress_into_shared(const uint64_t* GL_compressed_values, uint64_t* SH_decompressed_values, int PR_group_index, const int GROUP_SIZE, const int PREFIX_LEN, const int WORD_SIZE, /*debug*/ int64_t curr_block, int64_t curr_thread){
	
	//Step 0) cast shared memory to write data size (8 bit in this case) and intialize write index (used to write with)
	uint8_t* SH_decompressed_values_8bit = (uint8_t*)SH_decompressed_values;
	uint8_t PR_write_index = 0;
	
	//Step 1) calculate start_word/bit for this group
	int partialWord_Bits=0;
	int partialWord_firstEmptyBit = WORD_SIZE-1;
	int word_i = 0;
	uint64_t z1;
	uint8_t p1;
	int read_word_start;
	int read_position_start;
	int len1;
	for(int gi = 0; gi <= PR_group_index; gi++){
		if((partialWord_firstEmptyBit+1) >= (GROUP_SIZE + PREFIX_LEN)){		
			z1 = (uint64_t) ((GL_compressed_values[word_i] >> ((partialWord_firstEmptyBit+1)-GROUP_SIZE)) & ((uint64_t)(((uint64_t)1<<GROUP_SIZE)-1)));
			p1 = (uint8_t) ((GL_compressed_values[word_i] >> ((partialWord_firstEmptyBit+1)-(GROUP_SIZE+PREFIX_LEN))) & ((uint8_t)(pow(2,PREFIX_LEN)-1)));
			read_word_start= word_i;
			if(partialWord_firstEmptyBit == (GROUP_SIZE+PREFIX_LEN-1)) read_word_start++; //edge case (perfect)
		}else{ //partial Read
				uint64_t p_elem = (uint64_t) (GL_compressed_values[word_i]); //no shift since is in lower already
				uint8_t bits_first = partialWord_firstEmptyBit+1;
				uint8_t bits_second = (GROUP_SIZE+PREFIX_LEN)-bits_first;
				uint64_t elem = (uint64_t) (GL_compressed_values[word_i+1] >> (WORD_SIZE-bits_second));
				
				//have the upper and lower bytes. Now to mask them and combine
				p_elem = p_elem & (uint64_t)(((uint64_t)1<<bits_first)-1);
			   	elem = elem & (uint64_t)(((uint64_t)1<<bits_second)-1);
			
				elem = (p_elem << (bits_second)) | elem;
				//now can shift read z1, p1

				z1 = (uint64_t) (elem >> PREFIX_LEN);
				p1 = (uint8_t) (elem & ((uint8_t)(pow(2,PREFIX_LEN)-1)));
				read_word_start= word_i+1;
		}
		read_position_start = (partialWord_firstEmptyBit-(GROUP_SIZE+PREFIX_LEN)+ WORD_SIZE) % WORD_SIZE;
		
		//count number of 1s
		uint8_t count =0;
		uint64_t z1_copy = z1;
		while(z1_copy){
			count++;
			z1_copy = z1_copy & (z1_copy-1);
		}
		len1 = GROUP_SIZE + PREFIX_LEN + count*(p1+1);
		
		int num_words = (len1+partialWord_Bits)/WORD_SIZE;
		partialWord_Bits = (len1+partialWord_Bits) % WORD_SIZE;
		partialWord_firstEmptyBit = (WORD_SIZE-1)-partialWord_Bits; 
		//word_i and partialWord_firstEmptyBit poit to next group start position
		word_i+= num_words;
	}
	
	//printf("(block=%d,thread=%d): read_word_start=%d, read_pos=%d, z1=%ld, p1=%d\n", (int) curr_block, (int) curr_thread, read_word_start, read_position_start, z1, (uint16_t)p1);

	//Step 2) Have all the neccesary information, now can decompress group and write to shared memory
	uint8_t sign_mask = (uint8_t) pow(2,p1);
	uint8_t mask = (uint8_t) (pow(2, p1)-1) | sign_mask;
	p1+=1;
	len1 -= (GROUP_SIZE + PREFIX_LEN);
	uint8_t elem = 0;
	uint8_t elem_i = 0;
	while(len1 >= 0 && elem_i<(GROUP_SIZE)){ //use != 0 b/c should be exact by defin
		uint64_t bit_mask = (uint64_t)(pow(2,GROUP_SIZE-1-elem_i));
		if(len1 == 0) assert((bit_mask&z1) == 0);
		if((bit_mask & z1) == 0){//do not do read here, just place a zero
			SH_decompressed_values_8bit[PR_write_index++] = 0x00;
			elem_i++;
			continue;
		}
		
		if((read_position_start - p1) < -1){ //partial read
			uint8_t p_elem = (uint8_t) (GL_compressed_values[read_word_start]); //no shift since is in lower already
			uint8_t bits_first = read_position_start+1;
			uint8_t bits_second = p1-bits_first;
			elem = (uint8_t) (GL_compressed_values[read_word_start+1] >> (WORD_SIZE-bits_second));			
			//have the upper and lower bytes. Now to mask them and combine
			p_elem = p_elem & (uint8_t)(((uint8_t)1<<bits_first)-1);
			elem = elem & (uint8_t)(((uint8_t)1<<bits_second)-1);
		
			elem = (p_elem << (bits_second)) | elem;
			read_word_start++;
			read_position_start -= p1;
			read_position_start += WORD_SIZE;
		}else{
			elem = (uint8_t) (GL_compressed_values[read_word_start] >> (read_position_start-p1+1)) & mask;
			read_position_start -= p1; 
			if(read_position_start == -1) { //perfect boundary
				read_position_start += WORD_SIZE;
				read_word_start++;
			}
		}
		
		if(sign_mask & elem){
			elem &= ~sign_mask; //remove it
			elem |= 0x80; //add it back to front
			
			elem = ~elem + 1; //2's comp
			//printf("(block=%d,thread=%d): PR_write_index=%d, elem_decompressed=%d, elemi=%ld\n", (int) curr_block, (int) curr_thread, PR_write_index, (int) static_cast<int8_t>(0x80 | elem), elem_i+curr_thread*GROUP_SIZE);

			SH_decompressed_values_8bit[PR_write_index++] = (0x80 | elem);
		}
		else{
			//printf("(block=%d,thread=%d): PR_write_index=%d, elem_decompressed=%d, elemi=%ld\n", (int) curr_block, (int) curr_thread, PR_write_index, (int) static_cast<int8_t>(elem), elem_i+curr_thread*GROUP_SIZE);
			SH_decompressed_values_8bit[PR_write_index++] = (elem);
		}
						
		elem_i++;
		len1-=p1;
	}
	assert(len1 == 0);
	assert(elem_i == GROUP_SIZE);
}

//This function is to be called by each thread, it handles one group
//Input: pointer to uncopressed values, pointer to compressed values region to write to, pointer to group_len vector, and pointer to group_word_index vector. All vectors should be pointing to exactly the position to write to
//Output: writes compressed words into shared memory and populates group_len and group_word_index 
__device__ void unit_compress_into_shared(uint8_t* SH_uncompressed_values, uint64_t* SH_compressed_values, uint16_t* SH_group_len, const int PR_non_padded_elemets_per_thread, const int GROUP_SIZE, const int PREFIX_LEN, const int DATA_SIZE, const int WORD_SIZE, /*debug*/ int64_t curr_block, int64_t curr_thread){
	//Step 0) cast shared memory to read data size (8 bit in this case) and intialize write index (used to read with)
	//cast shared memory to group_len size (16 bit enought) and word_index size (32 bit enough)
	uint8_t* SH_uncompressed_values_8bit = (uint8_t*)SH_uncompressed_values;
	uint16_t* SH_group_len_16bit = (uint16_t*)SH_group_len;
	uint8_t PR_uncompressed_write_index = 0; 
	uint8_t PR_compressed_write_index = 0; 
	
	//Step 1) do the basic shapeshifter algorithm to find prefix_bits/zero_vec_bits/non_zeros and write result back into shared mem
	uint64_t zero_vec_bits = 0x00;
	uint8_t zero_vec_size = GROUP_SIZE;
	uint8_t prefix_bits = 0x00; //do not count sign bit
	uint8_t prefix_size = PREFIX_LEN; //log(8) upto 7 (8-1sign bit)
	uint16_t num_bits_total = 0;
	
	uint8_t non_zeros = GROUP_SIZE;
	uint8_t max_num = 0x00;
	for(int j=0; j < GROUP_SIZE; j++){
		uint8_t v = 0;
		if(j < PR_non_padded_elemets_per_thread) 
			v=SH_uncompressed_values_8bit[j];
		//printf("(block=%d,thread=%d): uncomp_quantized_output=%d at index=%d\n", (int) curr_block, (int) curr_thread, (int) v, (int)curr_thread*GROUP_SIZE+j);
		
		if(v != 0) zero_vec_bits |= (uint64_t)(pow(2,GROUP_SIZE-1-j));
		else {
			non_zeros--;
			continue;
		}
		uint8_t sign = 0x00;
		uint8_t mag = 0x00;
		if((v & 0x80) == 0x80){//2's comp
			sign = 0x80;
			mag = ~(v-1);
		}else mag = v;
		
		if(mag > max_num && mag != pow(2,7)) { 
			max_num = mag;
			uint8_t cpy = max_num;
			int count = 0;
			while(cpy>0){
				count++;
				cpy= cpy >> 1;
			}
			prefix_bits = (uint8_t) count;
			//floor(log2(8)) == 2 !!!!!
			//prefix_bits = 1 + (uint8_t) floor(log2(max_num)); //required experimental flag
		}
		SH_uncompressed_values_8bit[PR_uncompressed_write_index++] = (sign | mag);
	}
	//printf("(block=%d,thread=%d): max_num=%d, prefix_bits=%d\n", (int) curr_block, (int) curr_thread, (int) max_num, (int) prefix_bits);
	
	num_bits_total = GROUP_SIZE+PREFIX_LEN+non_zeros*(prefix_bits+1);
	//Have the sign and magnitude vector and all group data, now to pack
	uint32_t SIZE_GROUP_MAX = ((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE)/WORD_SIZE); //(16*8) + 16+3 fits in 64*3
	if(((GROUP_SIZE+PREFIX_LEN+GROUP_SIZE*DATA_SIZE) % WORD_SIZE) > 0) SIZE_GROUP_MAX++;
	uint64_t word_1 = ((uint64_t) zero_vec_bits << (WORD_SIZE-GROUP_SIZE)) | ((uint64_t)prefix_bits << (WORD_SIZE-GROUP_SIZE-PREFIX_LEN));
	int bits_stored_word = (zero_vec_size + prefix_size);	
	int first_empty_bit_pos = (WORD_SIZE-1)-bits_stored_word;
	
	bool WORD_DONE = false;
	uint64_t spillover_bits = 0;
	int spillover_size = 0;
	uint16_t word_index = 0;
	bool WORD_DIRTY = false;
	
	//edge case all 0 bits
	if(PR_uncompressed_write_index == 0){
		assert(zero_vec_bits == 0);
		assert(prefix_bits == 0);
		//printf("(block=%d,thread=%d): write_index=%d word=0x%016lx\n", (int) curr_block, (int) curr_thread, (int) PR_compressed_write_index, word_1);
		SH_compressed_values[PR_compressed_write_index++] = word_1;
		word_index++; 
	}
	for(int i =0; i < PR_uncompressed_write_index; i++){
		if(spillover_size != 0){
			assert(word_1 == 0);
			assert(first_empty_bit_pos == (WORD_SIZE-1));
			word_1 |= spillover_bits << ((first_empty_bit_pos- spillover_size)+1);
			first_empty_bit_pos -= (spillover_size);
			spillover_bits = 0;
			spillover_size = 0;
			WORD_DIRTY = true;
		}
		//printf("(block=%d,thread=%d): i=%d workingwith=%d\n", (int) curr_block, (int) curr_thread, (int) i, (int) SH_uncompressed_values_8bit[i]);	
		uint64_t shift_val = (((SH_uncompressed_values_8bit[i])&0x80) >> (DATA_SIZE-prefix_bits-1)) | (SH_uncompressed_values_8bit[i] & (uint8_t)(pow(2,prefix_bits)-1));
		if((first_empty_bit_pos - (prefix_bits)) < 0){ //out of space need to spill into next word
			//atleast 1 bit can be written, b/c otherwise the word be be done already
			uint8_t bits_to_write = first_empty_bit_pos+1;
			//lower bits spil
			spillover_size = (prefix_bits) - (first_empty_bit_pos);
			spillover_bits = shift_val & (uint8_t) (pow(2, spillover_size)-1);
			
			shift_val >>= spillover_size;
			
			word_1 |= shift_val; //no need to shift up as this must be lowest b/c word is done after this
			first_empty_bit_pos -= (bits_to_write);
			
			assert(first_empty_bit_pos == -1);
			WORD_DIRTY = true;
			WORD_DONE = true;
			assert(word_index != SIZE_GROUP_MAX-1); //cannot spill over on last word
		}
		else{
			//have space for all, write it in
			word_1 |= (shift_val << (first_empty_bit_pos-(prefix_bits))); //+1 included for sign bit
			first_empty_bit_pos -= (prefix_bits+1);
			
			WORD_DIRTY = true;
			if(first_empty_bit_pos == -1) WORD_DONE = true;
		}
		
		if(WORD_DONE){
			//printf("(block=%d,thread=%d): write_index=%d word=0x%016lx\n", (int) curr_block, (int) curr_thread, (int) PR_compressed_write_index, word_1);
			SH_compressed_values[PR_compressed_write_index++] = word_1;
			word_1 = 0;
			first_empty_bit_pos = (WORD_SIZE-1);
			WORD_DONE = false;
			WORD_DIRTY = false;
			word_index++;
			
			if(word_index == SIZE_GROUP_MAX){ //cannot write anymore words
				if(i!= (PR_uncompressed_write_index-1)){
					assert(0); //wrote last group and still have stuff left	
				}
				if(spillover_size != 0){
					assert(0); //cant spillover if wrote max words possible
				}
			}
		}
	}
	//edge case, spillover on last? 
	if(spillover_size != 0){
		assert(word_1 == 0);
		assert(first_empty_bit_pos == (WORD_SIZE-1));
		word_1 |= spillover_bits << ((first_empty_bit_pos- spillover_size)+1);
		first_empty_bit_pos -= (spillover_size);
		spillover_bits = 0;
		spillover_size = 0;
		WORD_DIRTY = true;
	}

	
	if(WORD_DIRTY && word_index < SIZE_GROUP_MAX){
		//printf("(block=%d,thread=%d): write_index=%d word=0x%016lx\n", (int) curr_block, (int) curr_thread, (int) PR_compressed_write_index, word_1);
		SH_compressed_values[PR_compressed_write_index++] = word_1;
		word_index++; //1, 2, 3
	}
	
	//printf("(block=%d,thread=%d): numbits for this group=%d\n", (int) curr_block, (int) curr_thread, (int) num_bits_total);
	*SH_group_len_16bit = num_bits_total;
	//Note: don't need this because placing words max_words apart
	//dont need group_word_index array
	//*SH_group_word_index_16bit = word_index;
}
//This function is called be each thread, it handles packing of 2 adjacent groups
//Input: index of first group, index of second group (2nd group is shifted uptowards 1st), third group index is needed to determine the length of group 2, group len vectors are needed, this timesend the pointer to entire vector (no need to position) same thing for the compressed values vector. The compressed values vector is read and written to at the same time. The threads should write disjoint sections to avoid conflict
//Output: the packed version of Groups 1/2 written back into the compressed values vector (inplace packing)
__device__ void unit_compressed_to_packed(uint64_t* SH_compressed_values, uint64_t* SH_group_len, int PR_group_index1, int PR_group_index2, int PR_group_index3, const int WORD_SIZE, const int MAX_WORDS_PER_GROUP){
	//Step 0) cast group_len to correct size and set write_index
	uint16_t* SH_group_len_16bit = (uint16_t*)SH_group_len;
	
	uint16_t PR_compressed_write_index = 0;
	
	//Step 2) Calculate len1 and len2 and all other variables needed for the algoithm
	int len1 =0;
	int len2 =0;
	for(int i = PR_group_index1; i < PR_group_index2; i++)
		len1+= SH_group_len_16bit[i];
	for(int i = PR_group_index2; i < PR_group_index3; i++)
		len2+= SH_group_len_16bit[i];
	
	uint16_t word_i= (uint16_t) MAX_WORDS_PER_GROUP*PR_group_index1;
	uint16_t word2_i= (uint16_t) MAX_WORDS_PER_GROUP*PR_group_index2;
	uint8_t partialWord_Bits=0;
	uint8_t partialWord_firstEmptyBit = WORD_SIZE-1;
	
	//Step 3) do the pack algithm for 2 groups, shift group2 up towards group1
	
	//group1 could be spread over multiple words, find which 1 and where the final position is
	int num_words = (len1+partialWord_Bits)/WORD_SIZE;
	partialWord_Bits = (len1+partialWord_Bits) % WORD_SIZE;
	partialWord_firstEmptyBit = (WORD_SIZE-1)-partialWord_Bits; //start of next word can be here
	word_i+=num_words;
	//next word should be positioned at word i+num_words and in that word its position is shifted to the position above
	
	//now do the shifting
	uint8_t bubble_size  = partialWord_firstEmptyBit+1;
	//edge case, if bubble size = 64, want it at wordi not wordi+1
	int word_shifts = word2_i -(word_i+1) + bubble_size/WORD_SIZE;

	//shift such that entire group_next is shifted up by bubble_size
	//step1 move to next word 
	int num_words2 = len2/WORD_SIZE;
	if(len2 % WORD_SIZE != 0) num_words2++;
	if(word_shifts > 0){
		for(int i = 0; i < num_words2; i++){
			PR_compressed_write_index = word_i+i+1-bubble_size/WORD_SIZE;
			SH_compressed_values[PR_compressed_write_index] = SH_compressed_values[word2_i+i];
		}
	}
	
	if(bubble_size == WORD_SIZE){ //perfect boundary (first empty = 63) previous word shifts deal with this 
		return;
	}
	
	//need to do bunch or Ands/Shifts to move group2 up
	uint8_t word1_firstEmpty = partialWord_firstEmptyBit;
	int temp_len2;
	for(int i = 0; i < num_words2; i++){
		//this is the number of bits in the next word of this group
		if((num_words2-i) > 1) temp_len2 = WORD_SIZE;
		else temp_len2 = (len2 - WORD_SIZE*i);
		
		uint64_t word1 = SH_compressed_values[word_i+i];
		uint64_t word2 = SH_compressed_values[word_i+i+1];
		uint64_t word2_shift = word2 >> ((WORD_SIZE-1)-word1_firstEmpty);
		
		//Ok to or here, bc word1 bubble is 0's, and word2 has 0's
		//in behind, otherwise would need to create word1 &=0XFFF<word2>
		word1 = word1 |	(word2_shift);
		PR_compressed_write_index = word_i+i;
		SH_compressed_values[PR_compressed_write_index]= word1;
		
		if( ((int)temp_len2-(word1_firstEmpty+1)) <= 0){
			PR_compressed_write_index = word_i+i+1;
			SH_compressed_values[PR_compressed_write_index] = 0;
			return; //done all fit in this word
		}
		word2_shift = word2 << (word1_firstEmpty+1);
		
		PR_compressed_write_index = word_i+i+1;
		SH_compressed_values[PR_compressed_write_index] = word2_shift;
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
