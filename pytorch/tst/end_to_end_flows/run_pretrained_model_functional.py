import os
import subprocess
import time
import sys

import torch
import torchvision

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
sys.path.append(os.path.join(repo_root, "pytorch/src"))
import CustomConvolution


def get_quantized_networks():
    reference_quantized_model = torchvision.models.quantization.googlenet(pretrained=True, quantize=True)
    custom_quantized_model = torchvision.models.quantization.googlenet(pretrained=True, quantize=True)

    return (reference_quantized_model, custom_quantized_model)

def replace_conv_layers(model, impl_to_use):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv_layers(module, impl_to_use)
        
        if type(module) not in [torch.nn.intrinsic.quantized.ConvReLU2d, torch.nn.quantized.Conv2d]:
            continue

        fuse_relu = (type(module) == torch.nn.intrinsic.quantized.ConvReLU2d)
        new = CustomConvolution.CustomConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            groups=module.groups,
            bias=module.bias,
            padding_mode=module.padding_mode,
            impl_to_use=impl_to_use,
            is_fused_with_relu=fuse_relu
        )
        setattr(model, n, new)

def load_dataset():
    print("Loading partial ImageNet dataset...")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_root_dir = os.path.join(repo_root, "pytorch/tst/end_to_end_flows/dataset/")
    imagenet_data = torchvision.datasets.ImageFolder(
        root=dataset_root_dir,
        transform=transform,
    )
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=1,
    )
    print("Finished loading dataset")

    # Read in the category/class names
    with open(os.path.join(dataset_root_dir, "classes.txt"), "r") as f:
        categories = [s.split(":")[1].strip() for s in f.readlines()]
    
    return (data_loader, categories)

def postprocess_network_output(output):
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    return torch.nn.functional.softmax(output[0], dim=0)

def parse_predictions(raw_predictions, categories):
    # Show top categories per image
    probabilities, class_indices = torch.topk(raw_predictions, 5)
    for probability, class_idx in zip(probabilities, class_indices):
        category_name = categories[class_idx]
        probability_percent = "{:.2f}".format(probability * 100)
        print(f"\t{category_name} -> {probability_percent}%")

def main():
    print("\nLoading reference and modified models")
    model1, model2 = get_quantized_networks()
    replace_conv_layers(model2, "shapeshifter_gpu")
    model2.load_state_dict(model1.state_dict())

    print("\nBeginning comparison test")
    data_loader, categories = load_dataset()
    max_test_cases_to_run = 10

    with torch.inference_mode():
        for i, data in zip(range(max_test_cases_to_run), data_loader):
            input, _ = data
            sample_fname, _ = data_loader.dataset.samples[i]

            print(f"Running comparison test using {sample_fname}:")

            predictions_original = postprocess_network_output(model1(input))
            print(f"[original]", end=None)
            parse_predictions(predictions_original, categories)

            predictions_modified = postprocess_network_output(model2(input))
            print(f"[modified]", end=None)
            parse_predictions(predictions_modified, categories)

            # Check that the predictions match
            atol = 1e-02
            rtol = 1e-05
            match = torch.allclose(predictions_original, predictions_modified, rtol=rtol, atol=atol)
            assert match, "Difference found between original model and modified model"
            print("Predictions match!")
    
    print(f"Predictions for all {max_test_cases_to_run} images match between the two versions of model")


if __name__ == "__main__":
    main()