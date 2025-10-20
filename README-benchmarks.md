# GPU Audio Benchmark Suite - Benchmark Reference

Reference documentation for benchmarks, parameters, and platform implementations.

## Overview

Tests GPU latency performance for real-time audio processing across different computational patterns and memory access scenarios.

## Buffer Layout Reference

"Track‑major" = each track's samples are contiguous (`trackIdx * bufferSize + sampleIdx`). "Sample‑major" = samples for all tracks at a given time index are adjacent (interleaved).

| Benchmark | Input Layout | Output Layout | Notes |
|-----------|--------------|---------------|-------|
| DataTransfer (`BenchmarkDataTransfer`) | Linear region sized by input ratio (no per-track structure) | Linear region sized by output ratio | Simulates DAW uploads/downloads without track semantics (`Benchmarks/DataTransferBenchmark.swift:83`, `.metal` kernel is a no-op). |
| gain / GainStats (`BenchmarkGain`, `BenchmarkGainStats`) | Track-major audio buffers (`Metal/kernels_benchmark_staging.metal:35-78`) | Track-major audio; GainStats also writes a `[mean, max]` pair per track (`Benchmarks/GainStatsBenchmark.swift:120-176`). |
| RandomMemoryRead (`BenchmarkRndMem`) | Sample memory is a flat pool; playhead table is per track | Track-major audio copy (`Metal/kernels_benchmark_staging.metal:231-250`, `Benchmarks/RandomMemoryReadBenchmark.swift:95-111`). |
| IIRFilter (`BenchmarkIIRBiquad`) | Track-major audio with per-track biquad state (`Metal/kernels_iir.metal:12-39`) | Track-major audio (`Benchmarks/IIRFilterBenchmark.swift:110-158`). |
| FFT1D (`BenchmarkFFT`) | Track-major real input (`Metal/kernels_benchmark_staging.metal:43-98`) | Track-major frequency bins with interleaved real/imag pairs (`Benchmarks/FFTBenchmark.swift:79-133`). |
| Conv1D (`BenchmarkConv1D`) | Track-major audio and IR buffers (`Metal/kernels_benchmark_staging.metal:235-256`) | Sample-major (interleaved) output for coalesced writes (`Metal/kernels_benchmark_staging.metal:247-259`, `Benchmarks/Convolution1DBenchmark.swift:150-174`). |
| Conv1D_accel (`Convolution1DAccelBenchmark`) | Track-major audio/IRs; intermediate FFT buffers are per track | Sample-major (interleaved) output to match GPU FFT pipeline (`Benchmarks/Convolution1DAccelBenchmark.swift:200-213`). |
| ModalFilterBank (`BenchmarkModalFilterBank`) | Mode-major parameter buffer (8 floats per mode) (`Metal/kernels_benchmark_staging.metal:154-168`) | Track-major accumulation across output tracks (`Metal/kernels_benchmark_staging.metal:214-219`). |
| DWG1DNaive / DWG1DAccel | Track-major excitation/input buffers (`Benchmarks/DWG1DNaiveBenchmark.swift:104-149`) | Track-major waveguide outputs (`Benchmarks/DWG1DNaiveBenchmark.swift:332-358`, accel variant mirrors this). |
| FDTD3D (`fdtd3d_*`) | 3D pressure/velocity grids with staggered Yee layout; audio input buffer is track-major (`Benchmarks/FDTD3DBenchmark.swift:199-227`) | Track-major audio extracted from receiver probe (`Metal/kernels_fdtd3d.metal:187-211`, `Benchmarks/FDTD3DBenchmark.swift:369-372`). |

## Benchmark Categories

### Data Transfer Benchmarks

GPU memory transfer overhead with varying input/output ratios.

#### datacopy0199
1% input, 99% output. Tests output-heavy scenarios (synthesis, heavy processing).

#### datacopy2080
20% input, 80% output. Output-biased processing.

#### datacopy5050
50% input, 50% output. Balanced input/output.

#### datacopy8020
80% input, 20% output. Input-heavy scenarios (analysis, compression).

#### datacopy9901
99% input, 1% output. Input-dominant with minimal output.

All datacopy variants use standard buffer/track parameters. Ratios are hardcoded per variant.

### Basic Audio Processing Benchmarks

#### NoOp
Kernel launch overhead without computation. Standard parameters.

#### gain
Simple gain processing (multiply by 2.0). Tests basic arithmetic. Standard parameters.

#### GainStats
Gain processing with per-track statistics (mean, max). Standard parameters.

### Digital Signal Processing Benchmarks

#### IIRFilter
2nd order biquad filter with 2 state variables per track. Configurable coefficients, persistent state. Standard parameters.

#### Conv1D
1D convolution with windowed sinc impulse responses. Per-track IRs with frequency variation. Tests constant/texture memory performance vs device memory.
- `impulseResponseLength`: IR length (default: 256)
- `useConstantMemory`: Use constant memory for IRs (default: true)

#### Conv1D_accel
FFT-based convolution. Platform-specific optimizations. Standard parameters.

#### FFT1D
Real-to-complex FFT. Track-major input, interleaved real/imag output. Standard parameters.

### Memory Access Pattern Benchmarks

#### RndMemRead
Random memory access simulating granular synthesis. Non-coalesced memory patterns with per-track playhead management. Large sample pool tests cache behavior.
- `sampleMemorySize`: Sample pool size (default: ~128M samples, ~512MB)
- `minLoopLength`: Min loop (default: 1000 samples)
- `maxLoopLength`: Max loop (default: 48000 samples)

### Synthesis Algorithm Benchmarks

#### ModalFilterBank
Modal synthesis filter bank with complex number processing. Each mode has 8 parameters (amplitude, frequency, phase, state). Tests atomic operations for accumulation across modes.
- `numModes`: Filter bank size (default: 1024 × trackCount, max: 1M)
- `outputTracks`: Output channels (default: min(trackCount, 32))

#### DWG1DNaive
Digital waveguide synthesis with straightforward GPU parallelization. Per-waveguide control of length, tap positions, gain, reflection, damping.
- `minLength`/`maxLength`: Waveguide length range

#### DWG1DAccel
Optimized digital waveguide synthesis with platform-specific acceleration. Standard parameters.

#### FDTD3D
3D finite difference time domain room acoustics. Pressure-velocity formulation with staggered Yee grid, 7-point spatial stencil, leapfrog time-stepping. Time step computed for CFL stability. Memory scales O(roomSize³).
- `roomSize`: Grid dimensions (default: 50×50×50, WebGPU uses smaller defaults)
- `absorptionCoeff`: Wall absorption (default: 0.2)
- `soundSpeed`: Speed of sound m/s (default: 343.0)
- `spatialStep`: Grid spacing meters (default: 0.01)

## Standard Parameters

- **Buffer Size** (`--buffersize`): Samples per buffer (default: 512)
- **Track Count** (`--ntracks`): Parallel audio tracks (default: 128)
- **Sample Rate** (`--fs`): Hz (default: 48000)
- **Run Count** (`--nruns`): Benchmark iterations (default: 100)
- **DAW Simulation** (`--dawsim`): DAW-like data patterns (Metal-Swift)

## Platform Implementation Status

| Benchmark | Metal-Swift | CUDA | WebGPU | Notes |
|-----------|-------------|------|--------|-------|
| NoOp | ✓ | ✗ | ✓ | Kernel launch overhead |
| datacopy* | ✓ | ✗ | ✓ | Data transfer variations (5 ratios) |
| gain | ✓ | ✓ | ✓ | Basic gain processing |
| GainStats | ✓ | ✗ | ✓ | Gain with statistics |
| IIRFilter | ✓ | ✗ | ✓ | Biquad filtering |
| Conv1D | ✓ | ✓ | ✓ | Direct convolution |
| Conv1D_accel | ✓ | ✗ | ✓ | FFT-based convolution |
| FFT1D | ✓ | ✗ | ✓ | Real-to-complex FFT |
| RndMemRead | ✓ | ✓ | ✓ | Random memory access |
| ModalFilterBank | ✓ | ✓ | ✓ | Modal synthesis |
| DWG1DNaive | ✓ | ✗ | ✓ | Digital waveguides |
| DWG1DAccel | ✓ | ✗ | ✓ | Optimized waveguides |
| FDTD3D | ✓ | ✓ | ✓ | 3D room acoustics |

## Running Benchmarks

### Metal-Swift (macOS)
```bash
cd metal-swift && ./build.sh
./build/MetalSwiftBench --benchmarkFilter gain,Conv1D --buffersize 512 --ntracks 128
```

### CUDA (Linux/Windows)
```bash
cd cuda && make
./gpubench --benchmark gain --bufferSize 512 --nTracks 128
```

### WebGPU (Browser)
```bash
cd webgpu && python -m http.server 8000
# Open http://localhost:8000 in browser
```

## Output and Verification

All benchmarks measure kernel execution latency (median, p95, max) and verify against CPU reference implementations. Error metrics include max and mean error vs reference. Results output as JSON (`--json`, `--outputfile`).
