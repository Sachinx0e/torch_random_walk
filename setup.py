import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_extension():

    # get the sources
    sources = [
        "csrc/cpu/rw.cpp"
    ]
    
    extra_compile_args = {'cxx': ['-O2']}
    extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
    extra_compile_args['cxx'] += ['-fopenmp']

    extension = CppExtension(
        'torch_rw_native',
        sources,
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