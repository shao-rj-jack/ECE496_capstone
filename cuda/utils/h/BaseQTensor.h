#pragma once

// Import the appropriate PyTorch headers depending on the compilation flow being used
#ifdef COMPILE_THROUGH_PYTORCH
#include <torch/extension.h>
#else
#include <torch/torch.h>
#endif
#include <c10/core/Device.h>
#include <vector>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

class BaseQTensor {

public:
    BaseQTensor(const torch::Tensor& tensor);

    BaseQTensor(
        const std::vector<int64_t>& dims,
        c10::QScheme qscheme,
        std::vector<double> q_scale,
        std::vector<int64_t> q_zero_point,
        c10::ScalarType dtype,
        c10::Device device
    );

    virtual ~BaseQTensor() = default;

    /**
     * Mimics the sizes() method of torch::Tensor
     * 
     * Returns an IntArrayRef of the dimensions of the underlying tensor
    */
    c10::IntArrayRef sizes() const;

    /**
     * Mimics the dtype() method of torch::Tensor
     * 
     * Returns the data type of the underlying tensor
    */
    c10::ScalarType dtype() const;

    /**
     * Mimics the device() method of torch::Tensor
     * 
     * Returns the device where the underlying data buffer resides
    */
    c10::Device device() const;

    /**
     * Mimics the is_contiguous() method of torch::Tensor
    */
    bool is_contiguous() const;

    /**
     * Mimics the qscheme() method of torch::Tensor
     * 
     * Returns the 'qscheme_' quantization parameter of the underlying tensor
    */
    c10::QScheme qscheme() const;

    /**
     * Mimics the q_scale() method of torch::Tensor
     * 
     * Returns the 'scale' quantization parameter of the underlying tensor
    */
    double q_scale() const;

    /**
     * Mimics the q_zero_point() method of torch::Tensor
     * 
     * Returns the 'zero_point' quantization parameter of the underlying tensor
    */
    int64_t q_zero_point() const;

    /**
     * Mimics the q_per_channel_scales() method of torch::Tensor
     * 
     * Returns the 'per-channel scale' quantization parameter of the underlying tensor
    */
    const std::vector<double>& q_per_channel_scales() const;

    /**
     * Mimics the q_per_channel_zero_points() method of torch::Tensor
     * 
     * Returns the 'per-channel zero_point' quantization parameter of the underlying tensor
    */
    const std::vector<int64_t>& q_per_channel_zero_points() const;

    /**
     * Returns a pointer to the underlying data buffer
     * 
     * Note: depending on usage, this buffer could reside in host memory or in GPU memory
    */
    HOST DEVICE virtual uint8_t* data() const = 0;

    /**
     * Returns the length (in bytes) of the underlying data buffer
    */
    HOST DEVICE virtual int64_t data_len() const = 0;

    virtual void cpu() = 0;

    virtual void cuda() = 0;

    /**
     * Returns a torch::Tensor with the same underlying values
    */
   virtual torch::Tensor toTorchTensor() const = 0;

protected:
    torch::Tensor getEmptyTorchTensor() const;

protected:
    /// Attributes pertaining to the raw (underlying) tensor
    std::vector<int64_t> dims_;
    c10::QScheme qscheme_;
    std::vector<double> q_scale_;       // One element for per-tensor, multiple elements for per-channel
    std::vector<int64_t> q_zero_point_; // One element for per-tensor, multiple elements for per-channel
    c10::ScalarType dtype_;

    // On which device the underlying tensor data is held
    c10::Device device_;
};