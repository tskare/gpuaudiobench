# Metal 

Metal implementation of the benchmarks for Mac M1+.

## Building and Running

The code has been placed on top of a modified Apple developer sample, so the
corresponding XCode project should compile directly on Intel or Apple Silicon systems with a
compatible Metal GPU.

You may specify `--help` for details. This produces output like the following:

```
Usage: ./GPUAudioBench --buffersize 512 --benchmark Conv1D --fs 48000 --ntracks 128Supported benchmarks:
  datacopy0199
  datacopy2080
  datacopy5050
  datacopy8020
  datacopy9901
  gain
  NoopKernelLaunch
  IIRFilter
  ModalFilterBank
  RndMemRead
  Conv1D
Optional flags:
  --dawsim: Simulate DAW-like behavior
  --dawsim_delay: Simulated buffer interarrival time in milliseconds
  --skip-human-readable-summary: Disable human-readable summary
  --nruns: Number of runs
  --help: Print this help message
  --outputfile: Output file for stats [currently set by benchmarks dynamically]
```

## Adjusting parameters
You may specify parameters on the commandline, or a greater set by modifying the code (surfacing all these in the console is forthcoming)

If working on one benchmark in specific you may wish to consider setting the default parameters at the top of `main.m`, for example:

`NSString *benchmarkName = @"Conv1D";`

this may be changed to another one of the benchmarks listed in `--help`

## License Notes

The XCode project used as a starting point the "Performing Calculations on a GPU" Apple Developer sample.

This was to set up any dependencies or any GPU-specific items, include relevant headers etc., and give users a supported target to debug any issues against. 
