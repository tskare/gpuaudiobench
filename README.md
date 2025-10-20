# GPGPU Audio Benchmark Suite

Microbenchmarks and domain-specific benchmarks to determine feasibility of a platform

The repository currently has multiple implementations that will compile independently:
- `cuda/` Windows+CUDA (should work on Linux, see note below)
- `metal-swift/` MacOS Apple Silicon (should work on Intel, see note below)
- `webgpu/` Cross-platform in-browser benchmarks built on the WebGPU API. Work in progress.

Note a platform passing these benchmarks is not a sufficient condition that a practical GPU-accelerated audio plugin will work out, nor are concerns here a guarantee that a platform is not suitable. Resource contention, OS scheduling bottlenecks and more can occur in real-time.

## What's New

### Version 2.0 Beta
- MacOS version rewritten in Swift
- Added experimental WebGPU platform
- Approximately doubled the number of benchmarks
- Attempted much more sharing of code between benchmarks
- Major cleanup; individual platforms are still completely independent but should share more structure and constants.
- New focus on practicality: essentially, per-platform and library optimizations are allowed and encouraged. Accelerated convolution versus large time-series straightforward convolutions.

### In progress

- Better documentation
- `webgpu/` improvements

## Project Overview

This project contains microbenchmarks and domain-specific benchmarks towards qualifying platforms for real-time GPGPU audio use. Benchmarks include data transfer patterns, audio processing algorithms (gain, IIR filters, convolution), modal synthesis, random memory access, and digital waveguide synthesis. 

Subdirectories:

- `cuda` for NVIDIA CUDA implementation (Windows/Linux).
- `metal-swift` for Apple Metal + Swift implementation (new target platform for Mac).
- `analysis` for analysis and utility scripts.
- `media` for artifacts such as the original DAFx24 poster

Directories are independent but checked out in the same repository. Each directory may contain its own `README.md` with details for that platform.

This project is GPL-3 licensed but please mail if this does not work for you. Please note any LICENSEs in `cuda` or `metal-swift` speficially if spinning off a project; these are from the IDE sample projects used as a base for increased compatibility on new developers' machines where
the relevant GPGPU toolkits are likely already installed.

We aim to require no dependencies beyond the relevant GPGPU toolkit and the C++ standard library. We aim to require minimal project configuration.

## Status

The code is expected to build on MacOS and Windows, with Linux supported but to be tested. In case of issues, please file an issue on GitHub or reach out to travisskare@gmail.com.

## AI Disclosure

Version 2 includes use of AI agents for assiting development and especially golden case validation and parameter documentation.

Additionally, the parameter documentation and such is temporarily all LLM-written. I had fallen behind writing it with a few updates. Better documentation is forthcoming once everything is fully baked and set. In the meantime please beware of hallucination there, although I review items before they're pushed.


## Test platforms:

- CUDA platform was tested on Windows, originally developed on a GTX 1080Ti and since moved to an RTX 4070 card.
- Metal-Swift is tested on Apple Silicon. I'll try to respond to bugfixes on Intel and need to find somewhere to host a CI target for it.
- webgpu should be cross-platform, tested on Chrome Stable channel at the time of pushes.

Code was tested on Linux at an earlier point, but not the current version (I reimaged that machine to Windows). I'm aiming to restore a proper Linux+CUDA machine soon.