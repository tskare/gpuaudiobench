# GPU Audio Benchmark Suite

This project contains microbenchmarks and domain-specific benchmarks towards qualify platforms for realtime GPGPU audio use. 

Subdirectories:

- `cuda` for CUDA implementation.
- `metal` for Objective-C Metal implementation. 
- `util` for any formatting and processing utility code as needed (none currently).

All three directories are independent.

Please note any LICENSEs in `cuda` or `metal` speficially if spinning off a project;
these are from the IDE sample projects used as a base for increased compatibility on new developers' machines where
the relevant GPGPU toolkits are likely already installed.

We aim to require no dependencies beyond the relevant GPGPU toolkit and the C++ standard library.

## Status

Various cleanup tasks are in progress, usually indicated by CLEANUP: or TODO: in the code.

The code is expected to build on MacOS and Windows, with Linux supported but to be tested. In case of issues, please file an issue on GitHub or reach out to travisskare@gmail.com.

## Test platforms:

CUDA platform was tested on Windows, GTX 1080Ti and RTX 4070 cards, with development and most test coverage on the 4070. Open for bug reports on Windows or Linux. I'll aim to add CI and binary releases on GitHub for these.
Code was tested on Linux at an earlier point, but not the current version (reimaged a machine to Windows). `cuda/Makefile` contains the Linux Makefile.

Metal was tested on Apple Silicon and Intel MacOS platforms, with development and most test coverage on Apple Silicon. Going forward, development will use Apple Silicon and its integrated GPU though I'm open to bug reports for Intel+Metal.
