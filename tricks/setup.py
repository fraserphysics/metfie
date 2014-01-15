#python3 setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(
    "temp_c",
    ["temp_c.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module]
)
