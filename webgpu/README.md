# WebGPU Audio Benchmark Suite

Browser-based implementation of the GPU Audio Benchmark Suite using WebGPU compute shaders. Provides real-time GPGPU audio processing benchmarks that run in the browser with feature parity to the Metal-Swift and CUDA implementations.

## Overview

- ES6 modules, no build step
- WGSL compute shaders
- Shoelace UI + Chart.js charts
- CPU reference validation where available

## Browser Support

WebGPU is supported in:

| Browser | Minimum Version | Notes |
| --- | --- | --- |
| Chrome | 113+ | Stable WebGPU support |
| Firefox | 121+ | Enable `dom.webgpu.enabled` in `about:config` on older builds |
| Safari | 18+ | Experimental support |

Enable WebGPU in Firefox: Set `dom.webgpu.enabled` to `true` in `about:config`

Power preference: the app requests `high-performance` in `requestAdapter`. On laptops, plug into power and disable battery-saver modes to avoid adapter throttling or device loss during long runs.

## Quick Start

1. **Clone and Navigate**:
   ```bash
   cd webgpu
   ```

2. **Serve Locally** (required for CORS):
   ```bash
   # Python 3
   python -m http.server 8000

   # Python 2
   python -m SimpleHTTPServer 8000

   # Node.js
   npx serve .

   # Or any other static file server
   ```

3. **Open Browser**:
   Navigate to `http://localhost:8000`

## Architecture

### Directory Structure
```
webgpu/
├── index.html              # Main UI entry point
├── css/
│   └── styles.css          # Custom styles
├── js/
│   ├── core/
│   │   ├── GPUABenchmark.js    # Base benchmark class
│   │   ├── BufferManager.js     # GPU buffer utilities
│   │   ├── Statistics.js        # Stats calculations
│   │   └── ParameterBuilder.js  # Parameter UI builder
│   ├── benchmarks/
│   │   ├── NoOpBenchmark.js            # Kernel launch overhead
│   │   ├── GainBenchmark.js            # Simple audio gain
│   │   ├── GainStatsBenchmark.js       # Gain with statistics
│   │   ├── DataTransferBenchmark.js    # Data transfer variants
│   │   ├── IIRFilterBenchmark.js       # Biquad digital filter
│   │   ├── Convolution1DBenchmark.js   # 1D convolution
│   │   ├── Convolution1DAccelBenchmark.js  # Optimized convolution
│   │   ├── FFTBenchmark.js             # Fast Fourier Transform
│   │   ├── ModalFilterBankBenchmark.js # Modal synthesis
│   │   ├── DWG1DNaiveBenchmark.js      # Digital waveguide (basic)
│   │   ├── DWG1DAccelBenchmark.js      # Digital waveguide (optimized)
│   │   ├── FDTD3DBenchmark.js          # 3D acoustic simulation
│   │   └── RandomMemoryBenchmark.js    # Random memory access
│   ├── shaders/
│   │   ├── *.wgsl                      # WebGPU compute shaders
│   └── app.js              # Main application logic
└── README.md
```

### Technology Stack

- **Language**: Plain JavaScript (ES6 modules)
- **GPU**: WebGPU compute shaders (WGSL)
- **UI**: Shoelace web components (CDN)
- **Charts**: Chart.js (CDN)
- **No Build**: Direct browser execution

## Benchmarks

Basic:
- NoOp
- Gain
- GainStats

Data transfer:
- datacopy0199
- datacopy2080
- datacopy5050
- datacopy8020
- datacopy9901

DSP:
- IIRFilter
- Conv1D
- Conv1D Accel
- FFT

Synthesis:
- ModalFilterBank
- DWG1D Naive
- DWG1D Accel
- FDTD3D

Memory:
- RandomMemory

## Configuration Options

Global settings apply to all benchmarks:

- **Buffer Size**: Audio buffer size in samples (64-4096, default: 512)
- **Track Count**: Number of parallel audio tracks (1-1024, default: 128)
- **Iterations**: Number of benchmark runs (10-1000, default: 100)
- **Warmup**: Warmup iterations to stabilize timing (0-20, default: 3)

Benchmark-specific parameters are available in the "Advanced Parameters" section when a benchmark is selected.

## Benchmark Results

Each benchmark provides:

### Performance Metrics
- **Median Latency**: Most representative timing (50th percentile)
- **95th Percentile (P95)**: Near-worst-case performance for real-time guarantees
- **Maximum**: Absolute worst case latency observed
- **Minimum**: Best case performance

### Quality Metrics
- **Validation**: GPU vs CPU output comparison (where applicable)
- **Error Analysis**: Maximum error, mean error, tolerance used
- **Samples Checked**: Number of samples validated
- **Pass/Fail Status**: Clear validation result

### Visualization
- **Latency Distribution**: Interactive histogram with Chart.js
- **Frequency Bins**: 30 bins showing distribution of latencies
- **Tooltips**: Percentage of samples in each bin
- **Y-axis Scaling**: Automatic scaling to show meaningful distribution

### Export Options
- **JSON Format**: Complete results with metadata
- **Browser Download**: Automatic save to download folder
- **Timestamp**: Filename includes ISO timestamp
- **Metadata**: Includes hardware info, configuration, validation results

## Browser Limitations & Differences from Native

WebGPU in browsers has some constraints compared to native Metal/CUDA implementations:

### Security & Safety
1. **No GPU Timing**: Browser security model prevents direct GPU timer access
   - Only CPU-side timing (includes transfer overhead)
   - Use `performance.now()` instead of GPU timestamps
   - Still accurate for comparing relative performance

2. **Memory Limits**: Browser tab memory restrictions
   - Large benchmarks (FDTD3D, Modal) use reduced defaults
   - Grid sizes clamped for stability (e.g., FDTD3D: 48 vs 64)
   - Sample pool sizes may be smaller than native

3. **Workgroup Size**: Conservative defaults for compatibility
   - Fixed at 64 threads for maximum browser compatibility
   - Native implementations can use 256+ on capable hardware

### Performance Characteristics
- **Additional Overhead**: JavaScript and browser runtime overhead
- **Transfer Costs**: CPU↔GPU transfers may be slower
- **Compilation**: WGSL shader compilation on first run (included in warmup)
- **Comparable Compute**: GPU compute performance similar to native once running

### Characteristics
- No installation required; runs in any WebGPU browser
- Same code runs on Windows/Mac/Linux
- Browser-based UI with interactive charts and real-time feedback
- Shareable via URL

## Implementation Details

### Base Class Pattern
```javascript
class GPUABenchmark {
  // Abstract methods for subclasses to implement
  async loadShader()        // Load WGSL compute shader from file
  async setupBuffers()      // Create GPU buffers and initialize data
  async performIteration()  // Execute one benchmark iteration
  async validate()          // Optional result validation against CPU reference

  // Concrete methods provided by base class
  async runBenchmark()      // Full benchmark execution with timing
  calculateStatistics()     // Statistical analysis of latencies
  exportResults()           // JSON export functionality
  cleanup()                 // Resource cleanup
}
```

### WebGPU Resource Management
- Automatic cleanup: Buffers and pipelines destroyed after use
- Memory tracking: BufferManager tracks allocation sizes and provides utilities
- Error handling: Error reporting with try-catch throughout
- Handles WebGPU initialization failures

### Shader Architecture
- Workgroup size: 64 threads (balance of compatibility and performance)
- Memory layout: Structure of Arrays
- Parameter passing: Uniform buffers for configuration, storage buffers for data
- Alignment and memory access patterns follow WGSL conventions

### Validation Strategy
All benchmarks (except NoOp) implement CPU reference validation:
1. Generate same input data for CPU and GPU
2. Run CPU reference implementation
3. Run GPU implementation
4. Compare outputs with appropriate tolerance
5. Report max error, mean error, and pass/fail status

## Performance Considerations

### WebGPU Configuration
- Workgroup size: 64 threads for compatibility, can be adjusted for specific hardware
- Buffer usage flags: Appropriate flags for each use case (STORAGE, COPY_DST, etc.)
- Command submission: Batched operations, single submit per iteration
- Memory transfer: GPU↔CPU transfers minimized where possible

### Timing Methodology
- Uses `performance.now()` for sub-millisecond timing
- `onSubmittedWorkDone()` synchronizes GPU work for measurement
- Warmup iterations eliminate shader compilation and cold-start effects
- Statistical analysis: Median and P95 for representative metrics

## Extending the Benchmark Suite

### Adding New Benchmarks

1. **Create WGSL Shader** (`js/shaders/your_benchmark.wgsl`):
   ```wgsl
   @group(0) @binding(0) var<storage, read> input: array<f32>;
   @group(0) @binding(1) var<storage, read_write> output: array<f32>;

   @compute @workgroup_size(64)
   fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
       // Your compute logic here
   }
   ```

2. **Implement Benchmark Class** (`js/benchmarks/YourBenchmark.js`):
   ```javascript
   import { GPUABenchmark } from '../core/GPUABenchmark.js';
   import { BufferManager } from '../core/BufferManager.js';

   export class YourBenchmark extends GPUABenchmark {
       constructor(device, bufferSize = 512, trackCount = 128) {
           super(device, 'YourBenchmark', bufferSize, trackCount);
           this.bufferManager = new BufferManager(device);
           this.referenceData = null;
       }

       async loadShader() { /* Load your WGSL shader */ }
       async setupBuffers() { /* Create GPU buffers, generate CPU reference */ }
       async performIteration() { /* Execute compute pass */ }
       async validate() { /* Compare GPU vs CPU */ }

       cleanup() {
           super.cleanup();
           this.bufferManager.destroyAll();
       }
   }
   ```

3. **Register in App** (`js/app.js`):
   - Add import statement
   - Add to `benchmarkCategories` with icon and description
   - Add to `benchmarkFactory` with factory function
   - Add parameters to `getBenchmarkParameters()` if needed

4. **Test**:
   - Verify shader loads without errors
   - Check validation passes
   - Confirm statistics are reasonable
   - Test with various configurations

### Common Patterns

Most audio benchmarks follow this pattern:
- **Input Buffer**: Audio samples (Float32Array)
- **Output Buffer**: Processed audio (Float32Array)
- **Parameter Buffer**: Configuration (uniform buffer, 16-byte aligned)
- **Workgroup Layout**: One workgroup per track, thread processes samples
- **CPU Reference**: Implement same algorithm in JavaScript for validation

## Debugging and Development

### Console Logging
Console output includes:
- WebGPU initialization status and hardware info
- Benchmark progress (every 10% for long runs)
- Buffer allocation tracking
- Validation results (pass/fail, errors)
- Error reporting with stack traces

### Browser DevTools
- **Performance Tab**: Profile GPU and CPU usage over time
- **Console**: Detailed logging, benchmark results, errors
- **Network Tab**: Monitor shader file loading
- **Memory Tab**: Track WebGPU resource usage and leaks

### Common Issues

**WebGPU not available**:
- Check browser version (Chrome 113+, Firefox 121+, Safari 18+)
- Firefox: Enable `dom.webgpu.enabled` in about:config
- Safari: Enable experimental WebGPU in Develop menu

**Shader compilation errors**:
- Check WGSL syntax in browser console
- Verify buffer bindings match shader layout
- Ensure proper structure alignment (16-byte for uniforms)

**Validation failures**:
- Check CPU reference implementation matches GPU algorithm
- Verify tolerance is appropriate for operation type
- Look for numerical precision issues

## Comparison with Other Implementations

| Feature | WebGPU | Metal-Swift | CUDA |
|---------|--------|-------------|------|
| **Platform** | Cross-platform (browser) | macOS only | NVIDIA GPUs only |
| **Language** | JavaScript | Swift | C++ |
| **Shaders** | WGSL | Metal Shading Language | CUDA C |
| **UI** | Web browser (interactive) | Command line | Command line |
| **Distribution** | URL/CDN | Binary executable | Binary executable |
| **Timing** | CPU-side only | GPU timestamps | GPU timestamps |
| **Benchmarks** | 17 (full parity) | 17 | 17 |
| **Validation** | ✓ All benchmarks | ✓ All benchmarks | ✓ All benchmarks |
| **Setup** | None (browser) | Build required | Build + CUDA toolkit |

### Performance Notes
- WebGPU adds browser overhead vs native implementations
- Suitable for cross-platform benchmarking and demos
- GPU compute performance comparable to native on current browsers
- Memory bandwidth limited by browser security model
- GPU compute typically within 10-20% of native performance

## Contributing

To add new benchmarks or features:

1. Follow existing code patterns and architecture
2. Test browser compatibility (Chrome, Firefox, Safari)
3. Add error handling with try-catch
4. Include CPU reference validation where applicable
5. Update documentation (this README, inline comments)
6. Follow STANDARD_PARAMS.md for default values
7. Test with various buffer sizes and track counts

## References

- **STANDARD_PARAMS.md**: Default parameters for all benchmarks
- **README-benchmarks.md**: Detailed algorithm descriptions
- **AGENTS.md**: Build commands and development workflow

## License

GPL-3 (same as parent project)

For alternative licensing options, contact: travisskare@gmail.com
