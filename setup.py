import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import glob
import os
import torch

def get_extension():

    # get the sources
    sources = glob.glob('csrc/**/*.cpp',recursive=True)
    sources_cuda = glob.glob('csrc/**/*.cu',recursive=True)   

    sources.extend(sources_cuda)

    # openmp
    extra_compile_args = {'cxx': ['-O2']}
    extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
    extra_compile_args['cxx'] += ['-fopenmp']

    # cuda
    define_macros = []
    define_macros += [('WITH_CUDA', None)]
    nvcc_flags = os.getenv('NVCC_FLAGS', '')
    nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
    nvcc_flags += ['-arch=sm_35', '-O2']
    extra_compile_args['nvcc'] = nvcc_flags

    # set include dirs
    include_dirs = ["csrc"]

    if torch.cuda.is_available() and torch.version.hip:
        # add rocrand headers
        include_dirs.extend([
            "/opt/rocm/include/rocrand",
            "/opt/rocm/include/hiprand"
        ])

    extension = CUDAExtension(
        'torch_rw_native',
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args
    )

    return extension

setuptools.setup(
    name="torch_rw",
    version="0.0.1",
    author="Sachin Gavali",
    author_email="sachinx0e@gmail.com",
    description="A pytorch extension library to perform random walks on graph",
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    ext_modules=[get_extension()],
    cmdclass={
        'build_ext': BuildExtension
    },
    test_requires=[
        'networkx==2.6.2',
        'pytest==6.2.4',
        'loguru==0.5.3'
    ]
)