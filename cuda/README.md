# CUDA

Contains Windows CUDA implementation of GPGPU audio benchmarks.

These should also run on Linux, but the included Makefile is experimental (repository will be updated).

## Structure

- `main.cu` has code to select a benchmark and run appropriate functions.
- `globals.cu[h]` contains constants and parameters shared across benchmarks.
- `bench*.cu[h]` files contain benchmark category implementations. The `cuh` file often contains parameters that may be adjusted.

## Status

The MacOS Metal version was the primary platform during development. The CUDA version has lagged being fully tidied up, but

- The single-file implementation submitted
- Various cleanup tasks are in progress.
- Parameters need to be moved to be commandline args versus requiring recompilation.
- Digital waveguide mesh / stenciling benchmark is in progress.

## Running (Windows)

A Visual Studio 2022 project is provided but hasn't been tested on a fresh install of Windows.

In case of error on your system, the code has been designed to be compiled with CUDA Toolkit sample projects:

1. Begin with any working CUDA toolkit sample.
1. Copy in the `.cu` and `.cuh` files, ensuring the cu files are built if needed ("Add Existing File" in Visual Studio does this automatically).
1. Adjust any parameters in the `.cuh` files as needed.
1. Build and run the commandline executable (default project in Visual Studio).

## Running (Linux)

The included Makefile is experimental and will be updated as we acquire a test system.
