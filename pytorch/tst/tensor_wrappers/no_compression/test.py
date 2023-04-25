import torch
import tensor_wrappers


if __name__ == "__main__":
    original_tensor = torch.quantize_per_tensor(
        torch.randn([16, 7, 32, 32]),
        scale=0.05,
        zero_point=0,
        dtype=torch.quint8
    )

    print("Starting to wrap")
    wrapped_tensor = tensor_wrappers.UncompressedQTensor(original_tensor)
    print("Wrapping complete")
    wrapped_tensor.cuda()
    wrapped_tensor.cpu()

    print("Starting to unwrap")
    unwrapped_tensor = wrapped_tensor.toTorchTensor()
    print("Unwrapping complete")

    assert torch.allclose(torch.dequantize(original_tensor), torch.dequantize(unwrapped_tensor)), "Wrap/Unwrap logic doesn't preserve data"
    print("Unwrapped data matches original")
