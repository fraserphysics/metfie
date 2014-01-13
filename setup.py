#python3 setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("C", ["C.pyx"])],
    depends = ['Scalar.py']
)

ext_module = Extension(
    "first_c",
    ["first_c.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    depends = ['first.py']
)
