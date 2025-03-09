from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "talib._ta_lib",
        ["talib/_ta_lib.pyx", "talib/_ta_lib.c"],
        libraries=["ta-lib"],
        include_dirs=["/usr/include"],
        library_dirs=["/usr/lib"]
    )
]

setup(
    name="talib",
    ext_modules=cythonize(extensions),
    packages=["talib"],
    package_dir={"talib": "."},
)
