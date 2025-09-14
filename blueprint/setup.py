from setuptools import setup, Extension
import numpy as np

extra_compile_args=['-O3','-march=native','-mtune=native','-fopenmp']
extra_link_args=['-fopenmp']

# Try to link GMP for fast big-int in C
libraries = ['gmp']

ext = Extension(
    'fastmod',
    sources=['fastmod.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries,
)

setup(
    name='fastmod',
    version='0.1',
    ext_modules=[ext],
)
