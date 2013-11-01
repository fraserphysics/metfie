#python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("C", ["C.pyx"])],
    depends = ['Scalar.py']
)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("first_c", ["first_c.pyx"])],
    depends = ['first.py']
)
