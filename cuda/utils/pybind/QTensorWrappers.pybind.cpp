#include "ShapeShifterCompressedQTensor.h"
#include "UncompressedQTensor.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<UncompressedQTensor>(m, "UncompressedQTensor")
        .def(py::init<const torch::Tensor&>())
        .def("size", &UncompressedQTensor::sizes, "Get size of tensor")
        .def("cuda", &UncompressedQTensor::cuda, "Move to cuda")
        .def("cpu", &UncompressedQTensor::cpu, "Move to cpu")
		.def("toTorchTensor", &UncompressedQTensor::toTorchTensor, "Convert UncompressedQTensor to torch::tensor");

    py::class_<ShapeShifter::ShapeShifterCompressedQTensor>(m, "ShapeShifterCompressedQTensor")
		.def(py::init<const torch::Tensor&, const ShapeShifter::CompressionParams&>())
        .def("size", &ShapeShifter::ShapeShifterCompressedQTensor::sizes, "Get size of (decompressed) tensor")
        .def("cuda", &ShapeShifter::ShapeShifterCompressedQTensor::cuda, "Move to cuda")
        .def("cpu", &ShapeShifter::ShapeShifterCompressedQTensor::cpu, "Move to cpu")
		.def("toTorchTensor", &ShapeShifter::ShapeShifterCompressedQTensor::toTorchTensor, "Decompress BaseCompressedQTensor to torch::tensor");
    py::class_<ShapeShifter::CompressionParams>(m, "ShapeShifterCompressionParams")
        .def(py::init<size_t>());
}
