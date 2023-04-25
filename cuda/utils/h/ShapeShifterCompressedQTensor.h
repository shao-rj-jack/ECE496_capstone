/**
 * Set of utilities pertaining to compression and decompression using ShapeShifter
 * 
 * These utilities are intended to be used in CPU/host code (e.g. for the initial,
 * one-time compression of the weights, or for compressing activations before
 * a sequence of convolutions and decompressing at the end)
 * )
*/

#pragma once

#include "BaseQTensor.h"

namespace ShapeShifter {

constexpr size_t DATA_SIZE = 8;
constexpr size_t PREFIX_LEN = 3;
constexpr size_t WORD_SIZE = 64;

struct CompressionParams {
    size_t group_size;
};

class ShapeShifterCompressedQTensor : public BaseQTensor {
public:
	ShapeShifterCompressedQTensor(const torch::Tensor& tensor, const CompressionParams& params);

	ShapeShifterCompressedQTensor(
		const std::vector<int64_t>& dims,
		c10::QScheme qscheme,
        std::vector<double> q_scale,
        std::vector<int64_t> q_zero_point,
        c10::ScalarType dtype,
        c10::Device device,
        uint8_t* compressed_data,
		int64_t compressed_len_bytes,
		int num_groups,
		const CompressionParams& params
	);

	ShapeShifterCompressedQTensor(const ShapeShifterCompressedQTensor& other);

	~ShapeShifterCompressedQTensor();

	HOST DEVICE uint8_t* data() const {
		return compressed_data_;
	}

	HOST DEVICE int64_t data_len() const {
		return compressed_len_bytes_;
	}

	virtual void cpu();

	virtual void cuda();

	virtual torch::Tensor toTorchTensor() const;

	HOST DEVICE CompressionParams get_compression_params() const;

	HOST DEVICE int get_num_groups() const;

private:
	// The actual (flattened) buffer containing all the compressed data (including group data and values)
    uint8_t* compressed_data_;
    int64_t compressed_len_bytes_;

	// Some metadata about the compressed data
	int num_groups_;
	
    CompressionParams compression_params_;
};

}
