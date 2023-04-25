#include "BaseQTensor.h"

#include <iostream>

BaseQTensor::BaseQTensor(const torch::Tensor& tensor)
    : dims_(tensor.sizes().vec()),
      qscheme_(tensor.qscheme()),
      dtype_(at::typeMetaToScalarType(tensor.dtype())),
      device_(tensor.device()) {

    if (dims_.size() != 4) { // Our kernels that use this class make this assumption
        throw std::invalid_argument("Expecting rank-4 tensors");
    }
    if (this->device() != torch::kCPU) {
        throw std::runtime_error("CPU-side ShapeShifter utilities can only be applied on CPU tensors");
    }
    if (tensor.dtype() != torch::kQInt8 && tensor.dtype() != torch::kQUInt8) {
        throw std::invalid_argument("Expected qint8 or quint8 for tensor type");
    }
    if (this->qscheme_ != torch::kPerTensorAffine && this->qscheme_ != torch::kPerChannelAffine) {
        throw std::invalid_argument("Unsupported quantization scheme");
    }
    if (this->qscheme_ == torch::kPerChannelAffine && tensor.q_per_channel_axis() != 0) {
        throw std::invalid_argument("Expecting per-channel quantization axis to be dim-0");
    }

    if (this->qscheme_ == torch::kPerTensorAffine) {
        this->q_scale_ = {tensor.q_scale()};
        this->q_zero_point_ = {tensor.q_zero_point()};
    } else {
        torch::Tensor q_scales = tensor.q_per_channel_scales();
        torch::Tensor q_zero_points = tensor.q_per_channel_zero_points();
        auto q_scales_accessor = q_scales.accessor<double, 1>();
        auto q_zero_points_accessor = q_zero_points.accessor<int64_t, 1>();
        for (int i = 0; i < q_scales.numel(); i++) {
            this->q_scale_.push_back(q_scales_accessor[i]);
            this->q_zero_point_.push_back(q_zero_points_accessor[i]);
        }
    }
}

BaseQTensor::BaseQTensor(
    const std::vector<int64_t>& dims,
    c10::QScheme qscheme,
    std::vector<double> q_scale,
    std::vector<int64_t> q_zero_point,
    c10::ScalarType dtype,
    c10::Device device
) : dims_(dims),
    qscheme_(qscheme),
    q_scale_(q_scale),
    q_zero_point_(q_zero_point),
    dtype_(dtype),
    device_(device) {}

c10::IntArrayRef BaseQTensor::sizes() const {
    return c10::ArrayRef<int64_t>(this->dims_);
}

c10::ScalarType BaseQTensor::dtype() const {
    return this->dtype_;
}

c10::Device BaseQTensor::device() const {
    return this->device_;
}

bool BaseQTensor::is_contiguous() const {
    return true;
}

c10::QScheme BaseQTensor::qscheme() const {
    return this->qscheme_;
}

double BaseQTensor::q_scale() const {
    if (this->qscheme_ != torch::kPerTensorAffine) {
        throw std::runtime_error("Invalid invokation of q_scale()");
    }
    return this->q_scale_.at(0);
}

int64_t BaseQTensor::q_zero_point() const {
    if (this->qscheme_ != torch::kPerTensorAffine) {
        throw std::runtime_error("Invalid invokation of q_zero_point()");
    }
    return this->q_zero_point_.at(0);
}

const std::vector<double>& BaseQTensor::q_per_channel_scales() const {
    if (this->qscheme_ != torch::kPerChannelAffine) {
        throw std::runtime_error("Invalid invokation of q_per_channel_scales()");
    }
    return this->q_scale_;
}

const std::vector<int64_t>& BaseQTensor::q_per_channel_zero_points() const {
    if (this->qscheme_ != torch::kPerChannelAffine) {
        throw std::runtime_error("Invalid invokation of q_per_channel_zero_points()");
    }
    return this->q_zero_point_;
}

torch::Tensor BaseQTensor::getEmptyTorchTensor() const {
    if (this->qscheme_ == torch::kPerTensorAffine) {
        return torch::_empty_affine_quantized(
            this->sizes(),
            torch::TensorOptions()
                .dtype(this->dtype())
                .device(this->device()),
            this->q_scale(),
            this->q_zero_point()
        );
    } else {
        int64_t num_per_channel_params = static_cast<int64_t>(this->q_per_channel_scales().size());
        torch::Tensor per_channel_scales = torch::zeros({num_per_channel_params});
        torch::Tensor per_channel_zero_points = torch::zeros({num_per_channel_params});
        for (int64_t i = 0; i < num_per_channel_params; i++) {
            per_channel_scales.index_put_({i}, this->q_per_channel_scales()[i]);
            per_channel_zero_points.index_put_({i}, this->q_per_channel_zero_points()[i]);
        }
        return torch::_empty_per_channel_affine_quantized(
            this->sizes(),
            per_channel_scales,
            per_channel_zero_points,
            0,
            torch::TensorOptions()
                .dtype(this->dtype())
                .device(this->device())
        );
    }
}
