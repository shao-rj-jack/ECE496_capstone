/**
 * Set of utilities pertaining to converting between PyTorch's native tensor type and
 * the tensor wrapper type used by our custom kernels
 * 
 * These utilities are intended to be used in CPU/host code (e.g. for the initial,
 * one-time wrapping of the weights, or for wrapping activations before
 * a sequence of convolutions and unwrapping at the end)
 * )
*/

#pragma once

#include "BaseQTensor.h"

class UncompressedQTensor : public BaseQTensor {
public:
	UncompressedQTensor(const torch::Tensor& tensor);

	UncompressedQTensor(
		const std::vector<int64_t>& dims,
		c10::QScheme qscheme,
        std::vector<double> q_scale,
        std::vector<int64_t> q_zero_point,
        c10::ScalarType dtype,
        c10::Device device,
        uint8_t* data,
        int64_t len_bytes
	);

	UncompressedQTensor(const UncompressedQTensor& other);

	~UncompressedQTensor();

	HOST DEVICE uint8_t* data() const {
		return data_;
	}

	HOST DEVICE int64_t data_len() const {
		return len_bytes_;
	}

	virtual void cpu();

	virtual void cuda();

	virtual torch::Tensor toTorchTensor() const;

private:
	// The actual (flattened) buffer containing all the tensor data
    uint8_t* data_;
    int64_t len_bytes_;
};
