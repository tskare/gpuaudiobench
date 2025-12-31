# gpuudiobench (Metal+Swift)

Swift + Metal backend for the GPU Audio Benchmark suite.

## Layout

```
metal-swift/
├── MetalSwiftBench/
│   ├── Core/                    # Core protocols and infrastructure
│   │   ├── GPUABenchmark.swift # Base protocol and implementation
│   │   ├── BenchmarkRegistry.swift # Type-safe benchmark registry
│   │   └── Statistics.swift     # Statistical analysis
│   ├── Benchmarks/             # Benchmark implementations
│   ├── Metal/                  # Metal shaders and types
│   │   ├── ShaderTypes.h       # Shared types between Swift and Metal
│   │   └── kernels_*.metal     # Metal compute kernels
│   ├── Utilities/              # Helper utilities
│   └── main.swift              # Entry point
├── build.sh                    # Build script
└── README.md                   # This file
```

## Build

Requires Xcode + Xcode command line tools.

```bash
./build.sh
./build/MetalSwiftBench --benchmarkFilter gain
```

## Run with a regex filter

`./build/MetalSwiftBench --benchmarkFilter /copy(80|50)/,gain`

## Write CSV results for 200 runs
```bash
./build/MetalSwiftBench --benchmarkFilter gain,NoOp --nruns 200 --outputfile results.csv
```

## DAW simulation

Simulate per-buffer scheduling between iterations:

```bash
./build/MetalSwiftBench --benchmarkFilter gain --dawsim --dawsim-mode spin --dawsim-jitter-us 5
```

### Resources
- [Apple Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Introduction/Introduction.html)


## Notes
- Output formats: CSV, JSON
- Validation: CPU reference comparisons where available
- DAW simulation: `--dawsim` flags

## License

GPL-3.0 (same as parent project)
