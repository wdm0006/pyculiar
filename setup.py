from setuptools import Extension, setup

anomaly_module = Extension(
    "pyculiarity._cext.anomaly_module",
    sources=["pyculiarity/_cext/anomaly_module.c"],
    extra_compile_args=["-O3", "-std=c99"],
    libraries=["m"],
)

if __name__ == "__main__":
    setup(ext_modules=[anomaly_module])
