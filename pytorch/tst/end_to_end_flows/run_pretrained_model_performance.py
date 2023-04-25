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
    replace_conv_layers(model2, "baseline_gpu")
    model2.load_state_dict(model1.state_dict())

    print("\nBeginning comparison test")
    data_loader, categories = load_dataset()

    test_times = [0 for i in range(100)]
    with torch.inference_mode():
        for i, data in zip(range(101), data_loader):
            input, _ = data
            sample_fname, _ = data_loader.dataset.samples[i]

            if i == 0:
                print(f"Discarding first run...\n")
                postprocess_network_output(model1(input))
            else:
                print(f"===== TEST RUN #{i} using {sample_fname} =====")
                start_time = time.time()
                postprocess_network_output(model1(input))
                test_times[i - 1] = time.time() - start_time
                print(f"RUN #{i} took {test_times[i - 1]} seconds\n")
    
    total_time = 0
    for runtime in test_times:
        total_time += runtime

    print(f"AVERAGE TIME ACROSS ALL RUNS: {total_time / 100}\n")


if __name__ == "__main__":
    main()