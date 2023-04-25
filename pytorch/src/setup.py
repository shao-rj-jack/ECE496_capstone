import os
import subprocess

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

from setuptools import setup, Extension
from torch.utils import cpp_extension

def compile_cpp_extension(extension_name, cpp_source_rel_paths, cpp_header_rel_paths, is_cuda_extension):
    abs_to_rel = lambda rel_path_list : [os.path.abspath(os.path.join(os.path.dirname(__file__), x)) for x in rel_path_list]

    cpp_source_abs_paths = abs_to_rel(cpp_source_rel_paths)
    assert all([os.path.exists(x) for x in cpp_source_abs_paths]), "Cannot locate all source files"

    cpp_header_abs_paths = abs_to_rel(cpp_header_rel_paths) if cpp_header_rel_paths else []
    assert all([os.path.exists(x) for x in cpp_header_abs_paths]), "Cannot locate all header files"

    ext_module = cpp_extension.CUDAExtension if is_cuda_extension else cpp_extension.CppExtension
    setup(name=extension_name,
        ext_modules=[
            ext_module(
                name=extension_name,
                sources=cpp_source_abs_paths,
                extra_compile_args=
                    [f"-I{include_path}" for include_path in cpp_header_abs_paths] +
                    ["-DCOMPILE_THROUGH_PYTORCH"]
            )
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension}
    )


if __name__ == "__main__":
    # Use ccache to speed up compilations
    os.environ["CC"] = os.path.join(repo_root, '../ccache-4.7.4-linux-x86_64/g++')

    extensions_cpp_sources = {
        "dummy_conv_cpu": {
            "sources": [
                os.path.join(repo_root, 'cuda/dummy_cpu/src/dummy_convolution.cpp')
            ],
            "headers": [],
            "is_cuda_extension": False,
        },
        "baseline_conv_cpu": {
            "sources": [
                os.path.join(repo_root, 'cuda/baseline_cpu/src/convolution.cpp')
            ],
            "headers": [],
            "is_cuda_extension": False,
        },
        "baseline_conv_gpu": {
            "sources": [
                os.path.join(repo_root, 'cuda/baseline_gpu/src/convolution.cu'),
            ],
            "headers": [],
            "is_cuda_extension": True,
        },
        "shapeshifter_conv_gpu": {
            "sources": [
                os.path.join(repo_root, 'cuda/shapeshifter_gpu/src/convolution.cu'),
            ],
            "headers": [],
            "is_cuda_extension": True,
        },
        "tensor_wrappers": {
            "sources": [
                os.path.join(repo_root, 'cuda/utils/pybind/QTensorWrappers.pybind.cpp'),
            ],
            "headers": [],
            "is_cuda_extension": False,
        }
    }

    common_sources = [
        os.path.join(repo_root, 'cuda/utils/src/Conv2dMetadata.cpp'),
        os.path.join(repo_root, 'cuda/utils/src/BaseQTensor.cpp'),
        os.path.join(repo_root, 'cuda/utils/src/UncompressedQTensor.cpp'),
        os.path.join(repo_root, 'cuda/utils/src/ShapeShifterCompressedQTensor.cpp'),
    ]
    common_headers = [
        os.path.join(repo_root, 'cuda/utils/h/'),
    ]

    for ext_name, ext_params in extensions_cpp_sources.items():
        sources = ext_params["sources"] + common_sources
        headers = ext_params["headers"] + common_headers
        is_cuda_extension = ext_params["is_cuda_extension"]
        print(f"Compiling extension: {ext_name}")
        try:
            compile_cpp_extension(ext_name, sources, headers, is_cuda_extension)
        except Exception as e:
            print("Compilation failed")
            print(e)
            break