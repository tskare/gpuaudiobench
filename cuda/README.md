# CUDA GPU Audio Benchmark Suite

Windows/Linux CUDA implementation of real-time GPGPU audio processing benchmarks with feature parity to Metal-Swift implementation.

## Structure

- `main.cu` - Main entry point with command-line argument parsing and benchmark execution
- `globals.cu[h]` - Global constants, parameters, and shared utility functions
- `bench*.cu[h]` - Benchmark implementations built on the shared GPUABenchmark framework
- `bench_base.cu[h]` - Base class framework providing lifecycle, timing, and validation helpers
- `benchmark_utils.cuh` - Utility functions for memory management and statistics

## Implementation Status

CUDA implementation with parity to Metal-Swift:

- Benchmark framework with validation and statistics
- All 17 benchmarks implemented
- Runtime parameter configuration (buffer size, track count, sample rate)
- CSV and JSON output support
- Help system with benchmark descriptions
- Cross-platform build system (Windows Visual Studio + Linux Makefile)
- Statistical analysis with deadline tracking
- Minimal dependencies (CUDA Toolkit + cuFFT only, helper_cuda.h removed)

## Available Benchmarks

The suite includes 17 benchmarks across 5 categories:

**Data Transfer** (5): `datacopy0199`, `datacopy2080`, `datacopy5050`, `datacopy8020`, `datacopy9901`
**Basic Audio** (3): `NoOp`, `gain`, `GainStats`
**DSP** (5): `IIRFilter`, `Conv1D`, `Conv1D_accel`, `ModalFilterBank`, `FFT1D`
**Physical Modeling** (3): `DWG1DNaive`, `DWG1DAccel`, `FDTD3D`
**Memory** (1): `RndMemRead`

## Command-Line Options

```bash
gpubench [options]

Options:
  --help              Print help message with full benchmark descriptions
  --list              List all available benchmarks
  --benchmark [name]  Run specific benchmark
  --fs [rate]         Set sampling rate (default: 48000)
  --bufferSize [size] Set buffer size (default: 512)
  --nTracks [count]   Set number of tracks (default: 256)
  --nRuns [count]     Set number of iterations (default: 100)
  --outputfile [file] Save results to CSV file
  --json              Output results in JSON format
```

## Examples

```bash
# Run gain benchmark with default settings
gpubench --benchmark gain

# Run IIR filter with custom parameters
gpubench --benchmark IIRFilter --bufferSize 1024 --nTracks 128

# Output JSON results to file
gpubench --benchmark FFT1D --json --outputfile results.json

# Save CSV results for analysis
gpubench --benchmark gain --outputfile benchmark_results.csv

# List all available benchmarks
gpubench --list
```

## Dependencies

**Required**:
- **CUDA Toolkit 11.0+**: Core CUDA runtime and development tools
- **cuFFT**: CUDA FFT library (included with CUDA Toolkit) - required for FFT1D and Conv1D_accel benchmarks

**Optional** (for older builds):
- NVIDIA's [cuda-samples](https://github.com/NVIDIA/cuda-samples) may be required for legacy builds
- Recent refactoring removed most helper_cuda.h dependencies

The Makefile will automatically check these locations if cuda-samples is needed:
- `CUDA_SAMPLES_DIR` environment variable
- `../../cuda-samples/Common` (sibling directory)
- Default CUDA Toolkit sample location

## Building and Running

### Linux

```bash
# Build with Makefile
cd cuda
make

# Run benchmark
./gpubench --benchmark gain
```

**Requirements**:
- CUDA Toolkit 11.0+
- gcc/g++ compatible with your CUDA version
- make

### Windows

**Visual Studio 2022 Project**:
1. Open `vectorAdd_vs2022.vcxproj` in Visual Studio
2. Verify CUDA Toolkit paths in project settings
3. Build solution (F7)
4. Run from command line or set command arguments in VS

**Manual Setup** (if project file fails):
1. Create new CUDA Runtime project in Visual Studio
2. Add all `.cu` and `.cuh` files to project
3. Configure include paths to point to cuda-samples
4. Build and run

**Requirements**:
- Visual Studio 2019/2022 with C++ and CUDA development tools
- CUDA Toolkit 11.0+
- Windows 10/11

