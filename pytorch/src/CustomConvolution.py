import math
import torch

# Imports for Python wrappers of custom C++ convolution implementations
import dummy_conv_cpu
import baseline_conv_cpu
import baseline_conv_gpu
import shapeshifter_conv_gpu

import tensor_wrappers

IMPL_TO_USE_MAP = {
    "dummy_cpu":    dummy_conv_cpu,
    "baseline_cpu": baseline_conv_cpu,
    "baseline_gpu": baseline_conv_gpu,
    "shapeshifter_gpu": shapeshifter_conv_gpu,
}

def is_shapeshifter_feasible_with_operands(input, weight, bias, stride, padding, group_size):
    _, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    stride = (stride, stride) if not (type(stride) is tuple) else stride
    stride_y, stride_x = stride
    padding_y, padding_x = CustomConv2dFunction._parse_padding(padding, weight, stride)
    out_height = 1 + (in_height + (2 * padding_y) - (kernel_height - 1) - 1) / stride_y
    out_width = 1 +  (in_width +  (2 * padding_x) - (kernel_width  - 1) - 1) / stride_x
    prefix_len = 3
    data_size = 8
    word_size = 64

    ## Weights
    num_weight_elems_decompresssed = out_channels * in_channels * kernel_height * kernel_width

    total_bytes_for_decompressed_weights = num_weight_elems_decompresssed               # 1 byte per element (int8_t)

    ## Inputs
    num_inputs_elems_needed_per_chunk_2d_slice = (kernel_height * in_width)
    num_input_groups_needed_per_chunk_2d_slice = math.ceil(num_inputs_elems_needed_per_chunk_2d_slice / group_size)
    num_input_groups_needed_per_chunk_2d_slice += 1
    num_input_groups_needed_per_chunk = in_channels * num_input_groups_needed_per_chunk_2d_slice

    total_bytes_for_decompressed_inputs = num_input_groups_needed_per_chunk * group_size * (data_size / 8)
    total_bytes_for_input_group_bookkeeping = num_input_groups_needed_per_chunk * 2     # 16 bits / 2 bytes per group index

    ## Outputs
    num_output_elems_per_block = out_channels * 1 * out_width;                          # 1 row per chunk
    num_output_groups_per_block = math.ceil(num_output_elems_per_block / group_size)
    max_words_per_compressed_output_group = math.ceil((group_size + prefix_len + (group_size * data_size)) / word_size)

    total_bytes_for_decompressed_outputs = num_output_elems_per_block                   # 1 byte per element (uint8_t)
    total_bytes_for_output_group_bookkeeping = num_output_groups_per_block * 2          # 2 bytes per group
    total_bytes_for_compressed_outputs = num_output_groups_per_block * max_words_per_compressed_output_group * (word_size / 8)
    total_bytes_other_output_bookkeeping = 8

    ## Total
    shared_mem_required = \
        total_bytes_for_decompressed_weights + \
        total_bytes_for_decompressed_inputs + \
        total_bytes_for_input_group_bookkeeping + \
        total_bytes_for_decompressed_outputs + \
        total_bytes_for_output_group_bookkeeping + \
        total_bytes_for_compressed_outputs + \
        total_bytes_other_output_bookkeeping

    max_supported_shared_memory_size = 99 * 1024
    return shared_mem_required <= max_supported_shared_memory_size

class CustomConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        impl_to_use,
        input,
        weights,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        scale=1.0,
        zero_point=0,
        is_fused_with_relu=False,
    ):
        assert impl_to_use, "Must specify which custom implementation of Conv2d to use"
        assert impl_to_use in IMPL_TO_USE_MAP.keys(), f"Unknown implementation type {impl_to_use}"
        custom_conv_module = IMPL_TO_USE_MAP[impl_to_use]

        assert dilation == 1, "Not supporting dilation"
        assert groups == 1, "Not supporting groups"

        # Tuplify the stride
        if not type(stride) is tuple:
            stride = (stride, stride)
        padding = CustomConv2dFunction._parse_padding(padding, weights, stride)
        # Tuplify the dilation
        if not type(dilation) is tuple:
            dilation = (dilation, dilation)

        if bias is None:
            bias = torch.zeros((weights.size()[0],), dtype=weights.dtype)

        stride = torch.tensor(stride)
        padding = torch.tensor(padding)

        assert isinstance(input, (tensor_wrappers.UncompressedQTensor, tensor_wrappers.ShapeShifterCompressedQTensor)), "Expecting input to be passed through tensor wrapper class"
        assert isinstance(weights, (tensor_wrappers.UncompressedQTensor, tensor_wrappers.ShapeShifterCompressedQTensor)), "Expecting weights to be passed through tensor wrapper class"

        run_on_gpu = impl_to_use.endswith("_gpu")
        if run_on_gpu:
            input.cuda()
            weights.cuda()
            bias = bias.cuda()
            stride = stride.cuda()
            padding = padding.cuda()
        
        # Assume multiple tensors returned from kernal. Concat tensors to single output tensor of same type
        if impl_to_use == "shapeshifter_gpu":
            outputs = custom_conv_module.forward(input, weights, bias, stride, padding, scale, zero_point, is_fused_with_relu)
            tensors = []
            for o in outputs:
                o.cpu()
                tensors.append(o.toTorchTensor())
            # Concat over the last dimension. Might have to change this.
            output = torch.cat(tensors, 2)
        else:
            output = custom_conv_module.forward(input, weights, bias, stride, padding, scale, zero_point, is_fused_with_relu)
            output.cpu()
            output = output.toTorchTensor()
        return output

    @staticmethod
    def backward(ctx, *args, **kwargs):
        assert 0, "Backward pass for CustomConv2d is unimplemented"

    @staticmethod
    def _parse_padding(padding, weights, stride=(1, 1)):
        if padding == 'valid':
            padding = (0, 0)
        elif padding == 'same':
            if stride != (1, 1):
                raise ValueError("'valid' padding requires a stride of 1")
            _, _, kernel_height, kernel_width = weights.size()
            # Compute the padding that will result in an output of the same
            # size as the input tensor
            padding_y = math.floor((kernel_height - 1) / 2)
            padding_x = math.floor((kernel_width - 1) / 2)
            padding = (padding_y, padding_x)
        elif not type(padding) is tuple:
            padding = (padding, padding)
        return padding


class CustomConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None,
        dtype=None,
        impl_to_use=None,
        is_fused_with_relu=False,
    ):
        super(CustomConv2d, self).__init__()

        assert padding_mode == "zeros", "Not supporting non-zero padding modes"

        # Save Conv2d parameters as object attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.impl_to_use = impl_to_use

        # Define the learnable parameters for Conv2d
        self.weight = torch.nn.Parameter(
            torch._empty_affine_quantized(
                (out_channels, int(in_channels / groups), kernel_size[0], kernel_size[1]),
                scale=1.0,
                zero_point=0,
                dtype=torch.qint8,
            ),
            requires_grad=False
        )
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_channels,), dtype=dtype),
                requires_grad=False
            )
        else:
            self.bias = None

        self.scale = float(1.0)
        self.zero_point = int(0)

        self.is_fused_with_relu = is_fused_with_relu

        if self.impl_to_use == "shapeshifter_gpu":
            self.group_size = 8
            self.shapeshifter_compression_params = tensor_wrappers.ShapeShifterCompressionParams(
                self.group_size
            )

        # Define initial values for the learnable parameters above
        # self.reset_parameters()

    # Resets learnable parameters to default values, as specified in
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html (see the 'Variables' section)
    def reset_parameters(self):
        val_range = self.groups / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        for param in self.parameters():
            param.data.uniform_(-val_range, +val_range)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.weight = torch.nn.Parameter(
            state_dict[prefix + 'weight'].clone(),
            requires_grad=False,
        )
        state_dict.pop(prefix + 'weight')
        self.bias = torch.nn.Parameter(
            state_dict[prefix + 'bias'].clone(),
            requires_grad=False,
        )
        state_dict.pop(prefix + 'bias')
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')
        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')
        
        use_shapeshifter = (self.impl_to_use == "shapeshifter_gpu")
        if use_shapeshifter:
            self.shapeshifter_wrapped_weight = tensor_wrappers.ShapeShifterCompressedQTensor(self.weight, self.shapeshifter_compression_params)
        self.no_comp_wrapped_weight = tensor_wrappers.UncompressedQTensor(self.weight)

    def forward(self, input):
        wrapped_input = None
        wrapped_weight = None
        final_decided_impl_to_use = None
        if self.impl_to_use == "shapeshifter_gpu":
            can_use_shapeshifter_kernel = is_shapeshifter_feasible_with_operands(input, self.weight, self.bias, self.stride, self.padding, self.group_size)
            if can_use_shapeshifter_kernel:
                final_decided_impl_to_use = "shapeshifter_gpu"
                wrapped_input = tensor_wrappers.ShapeShifterCompressedQTensor(input, self.shapeshifter_compression_params)
                wrapped_weight = self.shapeshifter_wrapped_weight
            else:
                final_decided_impl_to_use = "baseline_gpu"
                wrapped_input = tensor_wrappers.UncompressedQTensor(input)
                wrapped_weight = self.no_comp_wrapped_weight
        else:
            final_decided_impl_to_use = self.impl_to_use
            wrapped_input = tensor_wrappers.UncompressedQTensor(input)
            wrapped_weight = self.no_comp_wrapped_weight
        custom_output = CustomConv2dFunction.apply(final_decided_impl_to_use, wrapped_input, wrapped_weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.scale, self.zero_point, self.is_fused_with_relu)
        return custom_output

    def __repr__(self):
        name = "CustomConvReLU2d" if self.is_fused_with_relu else "CustomConv2d"
        return f'{name}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, scale={self.scale}, zero_point={self.zero_point}, padding={self.padding})'

    def __str__(self):
        name = "CustomConvReLU2d" if self.is_fused_with_relu else "CustomConv2d"
        return f'{name}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, scale={self.scale}, zero_point={self.zero_point}, padding={self.padding})'
