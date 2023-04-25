#include "ShapeShifterCompressedQTensor.h"

#include "c10/util/ArrayRef.h"
#include "cuda_runtime.h"
#include <bitset>
#include <cassert>
#include <cmath>
#include <vector>

using ShapeShifter::DATA_SIZE;
using ShapeShifter::PREFIX_LEN;
using ShapeShifter::WORD_SIZE;

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

namespace {

struct compression_return_type {
	std::vector<uint64_t> compressed_values;
	int num_groups;
	int total_words;
};

compression_return_type compression_impl(const uint8_t* raw_values, const size_t num_values, const ShapeShifter::CompressionParams& params) {
	const size_t GROUP_SIZE = params.group_size;

	if (GROUP_SIZE + PREFIX_LEN >= 64) {
		throw std::runtime_error("Parameters not as expected");
	}
	if (DATA_SIZE != 8 || PREFIX_LEN != 3) {
		throw std::runtime_error("Parameters not as expected");
	}

	std::vector<uint64_t> compressed_values;
	std::vector<int> group_word_index;
	std::vector<int> group_len;

	group_word_index.push_back(0); //group 0 starts at word 0
	const size_t num_groups = (num_values + GROUP_SIZE - 1) / GROUP_SIZE;
	for (size_t i = 0; i < num_groups; i++) {
		std::vector<uint8_t> group_vals;
		group_vals.reserve(GROUP_SIZE);
		for (size_t ii = 0; ii < GROUP_SIZE; ii++) {
			const size_t raw_val_idx = (i * GROUP_SIZE) + ii;
			if (raw_val_idx < num_values) {
				group_vals.push_back(raw_values[raw_val_idx]);
			} else {
				group_vals.push_back(0); // Padding of 0 at the end of a non-divisible-by-GROUP_SIZE input tensor
			}
		}
		
		uint64_t zero_vec_bits = 0x00;
		uint8_t zero_vec_size = GROUP_SIZE;
		uint8_t prefix_bits = 0x00; //do not count sign bit
		uint8_t prefix_size = PREFIX_LEN; //log(8) upto 7 (8-1sign bit)
	   	int num_bits_total = 0x00;
				
		std::vector<uint8_t> comp_values;
		comp_values.reserve(GROUP_SIZE);

		uint8_t non_zeros = GROUP_SIZE;
		uint8_t max_num = 0x00;
		for (size_t j = 0; j < GROUP_SIZE; j++) {
			uint8_t v = group_vals.at(j);
			if (v != 0) zero_vec_bits |= (uint64_t)(pow(2,GROUP_SIZE-1-j));
			else {
				non_zeros--;
				continue;
			}
			uint8_t sign = 0x00;
			uint8_t mag = 0x00;
			if((v & 0x80) == 0x80){//2's comp
				sign = 0x80;
				mag = ~(v-1);
				//if(v = pow(2, 7)); //-128 cannot represent using sign/mag in 8 bits
				//enocodes to -0 SPECIAL CASE
				//no point of increasing range here I think better to treat this seperatly? 
				
			} else mag = v;
			
			if(mag > max_num && mag != pow(2,7)) { //results in 9 bits if -128 case, hence the check for it, -0 -> 0 prefix bits, just store the sign
				max_num = mag;
				prefix_bits = 1 + (uint8_t) floor(log2(max_num));
			}
			comp_values.push_back(sign | mag);
		}
	
		//Have the sign and magnitude vector and all group data, now to pack
		num_bits_total = GROUP_SIZE + PREFIX_LEN + non_zeros*(prefix_bits + 1);
		const uint32_t SIZE_GROUP_MAX_BITS = GROUP_SIZE + PREFIX_LEN + (GROUP_SIZE * DATA_SIZE);
		const uint32_t SIZE_GROUP_MAX_WORDS = (SIZE_GROUP_MAX_BITS + WORD_SIZE - 1) / WORD_SIZE; //(16*8) + 16+3 fits in 64*3
		//NOTE::word_1 must be size WORD_SIZE
		uint64_t word_1 = ((uint64_t) zero_vec_bits << (WORD_SIZE-GROUP_SIZE)) | ((uint64_t)prefix_bits << (WORD_SIZE-GROUP_SIZE-PREFIX_LEN));
	   	size_t bits_stored_word = (zero_vec_size + prefix_size);	
		int first_empty_bit_pos = (WORD_SIZE - 1) - bits_stored_word;
		
		bool WORD_DONE = false;
		uint64_t spillover_bits = 0;
		size_t spillover_size = 0;
		size_t word_index = 0;
		bool WORD_DIRTY = false;
		
		//edge case all 0 bits
		if(comp_values.size() == 0){
			assert(zero_vec_bits == 0);
			assert(prefix_bits == 0);
			//printf("cpu: write_index=%d word=0x%016lx\n", (int) word_index, word_1);
			compressed_values.push_back(word_1);
			word_index++;
		}
		for (size_t i = 0; i < comp_values.size(); i++) {
			if (spillover_size != 0) {
				assert(word_1 == 0);
				assert(first_empty_bit_pos == (WORD_SIZE-1));
				word_1 |= spillover_bits << ((first_empty_bit_pos - spillover_size) + 1);
				first_empty_bit_pos -= (spillover_size);
				spillover_bits = 0;
				spillover_size = 0;
				WORD_DIRTY = true;
			}
			//printf("cpu: i=%d workingwith=%d\n", (int) i, (int) comp_values.at(i));	
			if ((first_empty_bit_pos - (prefix_bits)) < 0) { //out of space need to spill into next word
				//atleast 1 bit can be written, b/c otherwise the word be be done already
				uint64_t shift_val = ((comp_values.at(i)&0x80) >> (DATA_SIZE-prefix_bits-1)) | (comp_values.at(i) & (uint8_t)(pow(2,prefix_bits)-1));
				//printf("(ib=%d, sv=%s)", i, std::bitset<8>(shift_val).to_string().c_str());
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
			else {
				//have space for all, write it in
				uint64_t shift_val = ((comp_values.at(i)&0x80) >> (DATA_SIZE-prefix_bits-1)) | (comp_values.at(i)&(uint8_t)(pow(2,prefix_bits)-1));
				//printf("(i=%d, sv=%s)", i, std::bitset<8>(shift_val).to_string().c_str());
				
				word_1 |= (shift_val << (first_empty_bit_pos-(prefix_bits))); //+1 included for sign bit
				first_empty_bit_pos -= (prefix_bits+1);
				
				WORD_DIRTY = true;
				if(first_empty_bit_pos == -1) WORD_DONE = true;
			}
			
			if (WORD_DONE) {
				//printf("cpu: write_index=%d word=0x%016lx\n", (int) word_index, word_1);
				compressed_values.push_back(word_1);
				word_1 = 0;
				first_empty_bit_pos = (WORD_SIZE-1);
				WORD_DONE = false;
				WORD_DIRTY = false;
				word_index++;
				
				if (word_index == SIZE_GROUP_MAX_WORDS) { //cannot write anymore words
					if (i != (comp_values.size() - 1)) {
						assert(0); //wrote last group and still have stuff left	
					}
					if (spillover_size != 0) {
						assert(0); //cant spillover if wrote max words possible
					}
				}
			}
		}
		//edge case, spillover on last? 
		if (spillover_size != 0) {
			assert(word_1 == 0);
			assert(first_empty_bit_pos == (WORD_SIZE-1));
			word_1 |= spillover_bits << ((first_empty_bit_pos- spillover_size)+1);
			first_empty_bit_pos -= (spillover_size);
			spillover_bits = 0;
			spillover_size = 0;
			WORD_DIRTY = true;
		}

		if (WORD_DIRTY && word_index < SIZE_GROUP_MAX_WORDS) {
			//printf("cpu: write_index=%d word=0x%016lx\n", (int) word_index, word_1);
			compressed_values.push_back(word_1); //push last word
			word_index++; //1, 2, 3
		}
		
		//next group will start this word
		group_word_index.push_back(group_word_index.back() + word_index);
		//printf("cpu:numbits for this group=%d\n", (int) num_bits_total);
		group_len.push_back(num_bits_total);
		group_vals.clear();
	}

	// --- No Packing--Compressed into Words ---

	int group_i = 0;
	int word_i = 0;
	int total_words = compressed_values.size();
	int total_groups = group_len.size();
	uint8_t partialWord_Bits = 0;
	uint8_t partialWord_firstEmptyBit = WORD_SIZE - 1;
	while (group_i != (total_groups - 1)) {
		int len1 = group_len.at(group_i);
		//group1 could be spread over multiple words, find which 1 and where the final position is
		int num_words = (len1 + partialWord_Bits) / WORD_SIZE;
		partialWord_Bits = (len1 + partialWord_Bits) % WORD_SIZE;
		partialWord_firstEmptyBit = (WORD_SIZE - 1) - partialWord_Bits; //start of next word can be here
		word_i += num_words;
		//next word should be positioned at word i+num_words and in that word its position is shifted to the position above
		
		//now do the shifting
		uint8_t bubble_size = partialWord_firstEmptyBit+1;
		int word2_i = group_word_index.at(group_i+1);
		//edge case, if bubble size = 64, want it at wordi not wordi+1
		int word_shifts = word2_i -(word_i + 1) + bubble_size/WORD_SIZE;
		int len2 = group_len.at(group_i + 1);

		//shift such that entire group_next is shifted up by bubble_size
		//step1 move to next word 
		int num_words2 = len2 / WORD_SIZE;
		if (len2 % WORD_SIZE != 0) num_words2++;
		if (word_shifts > 0) {
			assert((word2_i+num_words2-1) < compressed_values.size());
			for (int i = 0; i < num_words2; i++) {
				compressed_values.at(word_i+i+1-bubble_size/WORD_SIZE) = compressed_values.at(word2_i+i);
			}
		}
		
		if(bubble_size == WORD_SIZE) { //perfect boundary (first empty = 63) previous word shifts deal with this 
			group_i++;
			continue;	
		}
		
		//need to do bunch or Ands/Shifts to move group2 up
		uint8_t word1_firstEmpty = partialWord_firstEmptyBit;
		int temp_len2;
		for (int i = 0; i < num_words2; i++) {
			//this is the number of bits in the next word of this group
			if((num_words2-i) > 1) temp_len2 = WORD_SIZE;
			else temp_len2 = (len2 - WORD_SIZE*i);

			uint64_t word1 = compressed_values.at(word_i+i);
			uint64_t word2 = compressed_values.at(word_i+i+1);
			uint64_t word2_shift = word2 >> ((WORD_SIZE-1)-word1_firstEmpty);
			
			//Ok to or here, bc word1 bubble is 0's, and word2 has 0's
			//in behind, otherwise would need to create word1 &=0XFFF<word2>
			word1 = word1 |	(word2_shift);
			compressed_values.at(word_i+i)= word1;
			
			if (((int)temp_len2-(word1_firstEmpty+1)) <= 0) {
				compressed_values.at(word_i+i+1) = 0;
				break; //done all fit in this word
			}
			word2_shift = word2 << (word1_firstEmpty+1);
			
			compressed_values.at(word_i+i+1) = word2_shift;
		}
		group_i++;
	}	
	int size = 0;
	for (auto s : group_len) {
		size += s;
	}
	int compressed_words = size/WORD_SIZE;
	if(size % WORD_SIZE) compressed_words++;
	compressed_values.resize(compressed_words);

	// --- Packed-Compressed into Words ---
	
	compression_return_type C_;
	C_.compressed_values = compressed_values;
	C_.num_groups = total_groups;
	C_.total_words = compressed_words;
	return C_;
}

std::vector<uint8_t> decompress_impl(const uint64_t* compressed_values, const size_t total_words, const int total_groups, const ShapeShifter::CompressionParams& params) {
	const size_t GROUP_SIZE = params.group_size;
	std::vector<uint8_t> decompressed_values;
	size_t partialWord_Bits=0;
	size_t partialWord_firstEmptyBit = WORD_SIZE-1;
	size_t word_i = 0;
	
	/*for(int i = 0; i < total_words; i++){
		std::cout << std::bitset<64>(compressed_values[i]) << std::endl;
	}*/
	
	for(size_t gi = 0; gi < static_cast<size_t>(total_groups); gi++){
		uint64_t z1;
		uint8_t p1;
		int read_word_start;
		if((partialWord_firstEmptyBit+1) >= (GROUP_SIZE + PREFIX_LEN)){		
			z1 = (uint64_t) ((compressed_values[word_i] >> ((partialWord_firstEmptyBit+1)-GROUP_SIZE)) & ((uint64_t)(((uint64_t)1<<GROUP_SIZE)-1)));
			//edge case, large group size, pow() return double
			//which looses precision so either cast to int before
			//doing -1 or use << operator (I think much better)
			//no FP unit needed
			p1 = (uint8_t) ((compressed_values[word_i] >> ((partialWord_firstEmptyBit+1)-(GROUP_SIZE+PREFIX_LEN))) & ((uint8_t)(pow(2,PREFIX_LEN)-1)));
			read_word_start= word_i;
			if(partialWord_firstEmptyBit == (GROUP_SIZE+PREFIX_LEN-1)) read_word_start++; //edge case (perfect)
		}else{ //partial Read
				assert(word_i < (total_words-1));
				uint64_t p_elem = (uint64_t) (compressed_values[word_i]); //no shift since is in lower already
				uint8_t bits_first = partialWord_firstEmptyBit+1;
				uint8_t bits_second = (GROUP_SIZE+PREFIX_LEN)-bits_first;
				uint64_t elem = (uint64_t) (compressed_values[word_i+1] >> (WORD_SIZE-bits_second));
				
				//have the upper and lower bytes. Now to mask them and combine
				p_elem = p_elem & (uint64_t)(((uint64_t)1<<bits_first)-1);
			   	elem = elem & (uint64_t)(((uint64_t)1<<bits_second)-1);
			
				elem = (p_elem << (bits_second)) | elem;
				//now can shift read z1, p1

				z1 = (uint64_t) (elem >> PREFIX_LEN);
				p1 = (uint8_t) (elem & ((uint8_t)(pow(2,PREFIX_LEN)-1)));
				read_word_start= word_i+1;
				
		}
		int read_position_start = (partialWord_firstEmptyBit-(GROUP_SIZE+PREFIX_LEN)+ WORD_SIZE) % WORD_SIZE;
		
		//NOTE to prevent sending the group_len vector
		//can calculate this easily using z1p1 (did before)
		//int len1 = group_len.at(gi);
		uint8_t count =0;
		uint64_t z1_copy = z1;
		while(z1_copy){
			count++;
			z1_copy = z1_copy & (z1_copy-1);
		}
		int len1 = GROUP_SIZE+PREFIX_LEN + count*(p1+1);
		//std::cout << "group length of gi=, len1= " << gi << " " << len1 << std::endl;
		int num_words = (len1+partialWord_Bits)/WORD_SIZE;
		partialWord_Bits = (len1+partialWord_Bits) % WORD_SIZE;
		partialWord_firstEmptyBit = (WORD_SIZE-1)-partialWord_Bits; 
		//word_i and partialWord_firstEmptyBit poit to next group start position
		word_i+= num_words;
		
		size_t read_word_end = word_i;
		size_t read_position_end = partialWord_firstEmptyBit;
		
		//read in the next len1 -(GROUP_SIZE-PREFIX_LEN) bits
		len1 -= (GROUP_SIZE + PREFIX_LEN);
		//one elem = p1+1 bits
		uint8_t sign_mask = (uint8_t) pow(2,p1);
		uint8_t mask = (uint8_t) (pow(2, p1)-1) | sign_mask;
		p1+=1;
		uint8_t elem = 0;
		uint8_t elem_i = 0;
		//edge case must keep count of elemi b/c may be <len1><0000>
		//need to push those zeros even after len1 is done
		while(len1 >= 0 && elem_i<(GROUP_SIZE)){ //use != 0 b/c should be exact by defin
			assert(elem_i < GROUP_SIZE);
			uint64_t bit_mask = (uint64_t)(pow(2,GROUP_SIZE-1-elem_i));
			if(len1 == 0) assert((bit_mask&z1) == 0);
			if((bit_mask & z1) == 0){//do not do read here, just place a zero
				decompressed_values.push_back(0x00);
				elem_i++;
				continue;
			}
			
			if((read_position_start - p1) < -1){ //partial read
				uint8_t p_elem = (uint8_t) (compressed_values[read_word_start]); //no shift since is in lower already
				uint8_t bits_first = read_position_start+1;
				uint8_t bits_second = p1-bits_first;
				assert(read_word_start < (total_words-1));
				elem = (uint8_t) (compressed_values[read_word_start+1] >> (WORD_SIZE-bits_second));
				
				//have the upper and lower bytes. Now to mask them and combine
				p_elem = p_elem & (uint8_t)(pow(2,bits_first)-1);
			   	elem = elem & (uint8_t)(pow(2, bits_second)-1);
			
				elem = (p_elem << (bits_second)) | elem;
				read_word_start++;
				read_position_start -= p1;
				read_position_start += WORD_SIZE;
			}else{
				elem = (uint8_t) (compressed_values[read_word_start] >> (read_position_start-p1+1)) & mask;
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
				decompressed_values.push_back(0x80 | elem);
			}
			else{
				decompressed_values.push_back(elem);
			}
							
			elem_i++;
			len1-=p1;
		}
		
		assert(len1 == 0);
		assert(elem_i == GROUP_SIZE);
		assert(read_word_start == read_word_end);
		assert(read_position_start == read_position_end);
	}

	return decompressed_values;	
}

}

namespace ShapeShifter {

ShapeShifterCompressedQTensor::ShapeShifterCompressedQTensor(const torch::Tensor& tensor, const CompressionParams& params)
	: BaseQTensor(tensor), compression_params_(params) {
	uint8_t* data_ptr;
	if (tensor.dtype() == torch::kQInt8) {
		data_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::qint8>());
	} else {
		data_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::quint8>());
	}

    compression_return_type compressed_output = compression_impl(
		data_ptr,
		std::accumulate(tensor.sizes().begin(), tensor.sizes().end(), 1, std::multiplies<int64_t>()),
		params
	);

    // Copy compressed data from the std::vector's internal buffer into a new buffer that will be
    // owned by the BaseCompressedQTensor
    // TODO: remove this copying
    this->compressed_len_bytes_ = compressed_output.compressed_values.size() * sizeof(uint64_t);
    this->compressed_data_ = new uint8_t[this->compressed_len_bytes_];
    std::memcpy(this->compressed_data_, compressed_output.compressed_values.data(), this->compressed_len_bytes_);
	this->num_groups_ = compressed_output.num_groups;
}

ShapeShifterCompressedQTensor::ShapeShifterCompressedQTensor(
	const std::vector<int64_t>& dims,
	c10::QScheme qscheme,
	std::vector<double> q_scale,
	std::vector<int64_t> q_zero_point,
	c10::ScalarType dtype,
	c10::Device device,
	uint8_t* data,
    int64_t len_bytes,
	int num_groups,
	const CompressionParams& params
) : BaseQTensor(dims, qscheme, q_scale, q_zero_point, dtype, device),
	compressed_data_(data),
	compressed_len_bytes_(len_bytes),
	num_groups_(num_groups),
	compression_params_(params) {}

ShapeShifterCompressedQTensor::ShapeShifterCompressedQTensor(const ShapeShifterCompressedQTensor& other) : BaseQTensor(other) {
    if (!other.compressed_data_) {
        return;
    }
	// Deep copy the compressed data
    this->compressed_len_bytes_ = other.compressed_len_bytes_;
	this->num_groups_ = other.num_groups_;
	this->compression_params_ = other.compression_params_;
    if (this->device_ == c10::kCUDA) {
        cudaMalloc(&this->compressed_data_, this->compressed_len_bytes_);
		cudaCheckErrors("");
        cudaMemcpy(this->compressed_data_, other.compressed_data_, this->compressed_len_bytes_, cudaMemcpyDeviceToDevice);
		cudaCheckErrors("");
    } else {
        this->compressed_data_ = new uint8_t[this->compressed_len_bytes_];
        std::memcpy(this->compressed_data_, other.compressed_data_, this->compressed_len_bytes_);
    }
}

ShapeShifterCompressedQTensor::~ShapeShifterCompressedQTensor() {
    if (!this->compressed_data_) {
        return;
    }
    if (this->device_ == c10::kCUDA) {
        cudaFree(this->compressed_data_);
		cudaCheckErrors("");
    } else {
        delete[] this->compressed_data_;
    }
	// std::cerr << "Deleted " << this->compressed_data_ << std::endl;
    this->compressed_data_ = nullptr;
    this->compressed_len_bytes_ = 0;
}

void ShapeShifterCompressedQTensor::cuda() {
	if (!this->compressed_data_) {
        return;
    }
	if (this->device_ == c10::kCUDA) {
		return;
	}
	// Deep copy the compressed data
	uint8_t* new_compressed_data_block;
	cudaMalloc(&new_compressed_data_block, this->compressed_len_bytes_);
	cudaCheckErrors("");
	cudaMemcpy(new_compressed_data_block, this->compressed_data_, this->compressed_len_bytes_, cudaMemcpyHostToDevice);
	cudaCheckErrors("");

	delete[] this->compressed_data_;
	this->compressed_data_ = new_compressed_data_block;
	this->device_ = c10::kCUDA;
}

void ShapeShifterCompressedQTensor::cpu() {
	if (!this->compressed_data_) {
        return;
    }
	if (this->device_ == c10::kCPU) {
		return;
	}
	// Deep copy the compressed data
	uint8_t* new_compressed_data_block = new uint8_t[this->compressed_len_bytes_];

	cudaMemcpy(new_compressed_data_block, this->compressed_data_, this->compressed_len_bytes_, cudaMemcpyDeviceToHost);
	cudaCheckErrors("");

	cudaFree(this->compressed_data_);
	cudaCheckErrors("");
	this->compressed_data_ = new_compressed_data_block;
	this->device_ = c10::kCPU;
}

torch::Tensor ShapeShifterCompressedQTensor::toTorchTensor() const {
	if (this->device() != torch::kCPU) {
		throw std::invalid_argument("Require CPU tensor for decompression");
	}
	
    std::vector<uint8_t> decompressed_values = decompress_impl(
		reinterpret_cast<const uint64_t*>(this->compressed_data_),
		this->compressed_len_bytes_ / (WORD_SIZE/8),
		this->num_groups_,
		this->compression_params_
	);
    // In the case of non-divisible tensor sizes, the compression step may have introduced padding at the end of the compressed data buffer
    // In case this padding made its way back into the decompressed values, discard these values
    const size_t required_num_elems = std::accumulate(this->dims_.begin(), this->dims_.end(), 1, std::multiplies<int64_t>());
    if (decompressed_values.size() < required_num_elems) {
		throw std::length_error("Didn't get back the expected number of elements from compressed data block");
	}
    decompressed_values.resize(required_num_elems);

    // Convert the decompressed data (an std::vector) into a torch::Tensor
	// For now, this is yet another copy
	torch::Tensor decompressed_tensor = this->BaseQTensor::getEmptyTorchTensor();

	uint8_t* data_ptr;
	if (decompressed_tensor.dtype() == torch::kQInt8) {
		data_ptr = reinterpret_cast<uint8_t*>(decompressed_tensor.data_ptr<c10::qint8>());
	} else {
		data_ptr = reinterpret_cast<uint8_t*>(decompressed_tensor.data_ptr<c10::quint8>());
	}
	std::memcpy(data_ptr, decompressed_values.data(), decompressed_values.size() * sizeof(uint8_t));

	return decompressed_tensor;
}

HOST DEVICE CompressionParams ShapeShifterCompressedQTensor::get_compression_params() const {
	return this->compression_params_;
}

HOST DEVICE int ShapeShifterCompressedQTensor::get_num_groups() const {
	return this->num_groups_;
}

}
