from setuptools import setup, find_packages
from Cython.Build import cythonize

ext_modules = cythonize(
    "peteos/**/*.py",
    language_level="3str",
    compiler_directives={"annotation_typing": False},
)

setup(
    ext_modules=ext_modules,
    packages=find_packages(include=["peteos", "peteos.*"]),
    zip_safe=False,
)
