import sys
import torch
import tensor_wrappers


if __name__ == "__main__":
    original_tensor = torch.quantize_per_tensor(
        torch.randn([16, 7, 32, 32]),
        scale=0.05,
        zero_point=0,
        dtype=torch.quint8
    )

    compression_params = tensor_wrappers.ShapeShifterCompressionParams(
        11 # group_size
    )

    print("Starting compression")
    compressed_tensor = tensor_wrappers.ShapeShifterCompressedQTensor(original_tensor, compression_params)
    print("Compression complete")

    compressed_tensor.cuda()
    compressed_tensor.cpu()

    print("Starting decompression")
    decompressed_tensor = compressed_tensor.toTorchTensor()
    print("Decompression complete")

    assert torch.allclose(torch.dequantize(original_tensor), torch.dequantize(decompressed_tensor)), "Compression/Decompression logic doesn't preserve data"
    print("Decompressed data matches original")
