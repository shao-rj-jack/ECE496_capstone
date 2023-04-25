#include "UncompressedQTensor.h"

#include "c10/util/ArrayRef.h"
#include "cuda_runtime.h"
#include <cassert>

UncompressedQTensor::UncompressedQTensor(const torch::Tensor& tensor) : BaseQTensor(tensor) {
	assert(this->device() == torch::kCPU);
	assert(tensor.dtype() == torch::kQInt8 || tensor.dtype() == torch::kQUInt8);
	uint8_t* data_ptr;
	if (tensor.dtype() == torch::kQInt8) {
		data_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::qint8>());
	} else {
		data_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::quint8>());
	}

    // Copy data from the torch::Tensor's internal buffer into a new buffer that will be
    // owned by the BaseQTensor
    this->len_bytes_ = tensor.numel() * sizeof(uint8_t);
    this->data_ = new uint8_t[this->len_bytes_];
    std::memcpy(this->data_, data_ptr, this->len_bytes_);
}

UncompressedQTensor::UncompressedQTensor(
	const std::vector<int64_t>& dims,
	c10::QScheme qscheme,
	std::vector<double> q_scale,
	std::vector<int64_t> q_zero_point,
	c10::ScalarType dtype,
	c10::Device device,
	uint8_t* data,
    int64_t len_bytes
) : BaseQTensor(dims, qscheme, q_scale, q_zero_point, dtype, device),
	data_(data),
	len_bytes_(len_bytes) {}

UncompressedQTensor::UncompressedQTensor(const UncompressedQTensor& other) : BaseQTensor(other) {
    if (!other.data_) {
        return;
    }
	// Deep copy the interal data buffer
    this->len_bytes_ = other.len_bytes_;
    if (this->device_ == c10::kCUDA) {
        cudaMalloc(&this->data_, this->len_bytes_);
        cudaMemcpy(this->data_, other.data_, this->len_bytes_, cudaMemcpyDeviceToDevice);
    } else {
        this->data_ = new uint8_t[this->len_bytes_];
        std::memcpy(this->data_, other.data_, this->len_bytes_);
    }
}

UncompressedQTensor::~UncompressedQTensor() {
    if (!this->data_) {
        return;
    }
    if (this->device_ == c10::kCUDA) {
        cudaFree(this->data_);
    } else {
        delete[] this->data_;
    }
    this->data_ = nullptr;
    this->len_bytes_ = 0;
}

void UncompressedQTensor::cuda() {
	if (!this->data_) {
        return;
    }
	if (this->device_ == c10::kCUDA) {
		return;
	}
	// Deep copy the internal data buffer
	uint8_t* new_data_block;
	cudaMalloc(&new_data_block, this->len_bytes_);
	cudaMemcpy(new_data_block, this->data_, this->len_bytes_, cudaMemcpyHostToDevice);

	delete[] this->data_;
	this->data_ = new_data_block;
	this->device_ = c10::kCUDA;
}

void UncompressedQTensor::cpu() {
	if (!this->data_) {
        return;
    }
	if (this->device_ == c10::kCPU) {
		return;
	}
	// Deep copy the internal data buffer
	uint8_t* new_data_block = new uint8_t[this->len_bytes_];
	cudaMemcpy(new_data_block, this->data_, this->len_bytes_, cudaMemcpyDeviceToHost);

	cudaFree(this->data_);
	this->data_ = new_data_block;
	this->device_ = c10::kCPU;
}

torch::Tensor UncompressedQTensor::toTorchTensor() const {
	assert(this->device() == torch::kCPU);

	torch::Tensor tensor = this->BaseQTensor::getEmptyTorchTensor();

	uint8_t* data_ptr;
	if (tensor.dtype() == torch::kQInt8) {
		data_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::qint8>());
	} else {
		data_ptr = reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::quint8>());
	}
	std::memcpy(data_ptr, this->data_, this->len_bytes_);

	return tensor;
}
