import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension


def get_extension():

    # get the sources
    sources = [
        "csrc/cpu/rw.cpp"
    ]

    extension = CppExtension(
        'torch_rw',
        sources
    )

    return extension

setuptools.setup(
    name="torch_rw",
    version="0.0.1",
    author="Sachin Gavali",
    author_email="sachinx0e@gmail.com",
    description="A pytorch extension library to perform random walks on graph",
    ext_modules=[get_extension()],
    cmdclass={
        'build_ext': BuildExtension
    }
)