# CUDA

Contains Windows/Linux CUDA implementation of GPGPU audio benchmarks.

## Structure

- `main.cu` has code to select a benchmark and run appropriate functions.
- `globals.cu[h]` contains constants and parameters shared across benchmarks.
- `bench*.cu[h]` files contain benchmark category implementations. The `cuh` file often contains parameters that may be adjusted.

## Status

The MacOS Metal version was the primary platform during development. The CUDA version has lagged being fully tidied up, but:

- The single-file implementation submitted in the first revision has been replaced with a multi-file implementation.
- Various cleanup tasks are in progress.
- Some parameters need to be moved to be commandline args versus requiring recompilation. However benchmarks may be selected at runtime vs. requiring a recompilation as in the originally submitted version. `--help` is available for details.
- Digital waveguide mesh / stenciling benchmark is in progress.

## Dependencies

There is a single dependency, the CUDA toolkit samples. However, at least on Linux these are not installed to `CUDA_HOME` by default; you can install them from NVIDIA's [cuda-samples](https://github.com/NVIDIA/cuda-samples) GitHub repository.

You may add them to your include path; the Makefill will also check for `cuda-samples/` as a sibling directory to your checkout of this project.

On Windows I had this in my includes from the CUDA toolkit, however it's possible I had introduced some "magic" setup steps along the way. I will try to replicate on a fresh install of Windows the next time I create a VM or image a PC. Feedback welcome on this point; you may have to add `cuda-samples/Common/` to your includes path. 

## Running (Windows)

A Visual Studio 2022 project is provided. You may need to adjust the "Include Paths" to point to your CUDA headers.

In case of error on your system, the code has been designed to be compiled from the CUDA Toolkits folder without modification. Ignore our provided Visual Studio project for the following steps:

1. Begin with any working CUDA toolkit sample.
1. Copy in the `.cu` and `.cuh` files, ensuring the cu files are built if needed ("Add Existing File" in Visual Studio does this automatically).
1. Adjust any parameters in the `.cuh` files as needed.
1. Build and run the commandline executable (default project in Visual Studio).

## Running (Linux)

A Makefile is included.


