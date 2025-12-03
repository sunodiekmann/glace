from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
# Might need to change paths - Run in shell to check
# pkg-config --cflags opencv4
# pkg-config --libs opencv4

# Adjust these if pkg-config showed something different
opencv_inc_dir = "/usr/include/opencv4"
opencv_lib_dir = "/usr/lib/x86_64-linux-gnu"

if not os.path.isdir(opencv_inc_dir):
    print("Error: OpenCV include directory does not exist:", opencv_inc_dir)
    raise SystemExit(1)

if not os.path.isdir(opencv_lib_dir):
    print("Error: OpenCV library directory does not exist:", opencv_lib_dir)
    raise SystemExit(1)

print("Using OpenCV include dir:", opencv_inc_dir)
print("Using OpenCV library dir:", opencv_lib_dir)

setup(
    name="dsacstar",
    ext_modules=[
        CppExtension(
            name="dsacstar",
            sources=["dsacstar.cpp", "thread_rand.cpp"],
            include_dirs=[opencv_inc_dir],
            library_dirs=[opencv_lib_dir],
            libraries=["opencv_core", "opencv_calib3d"],
            extra_compile_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
