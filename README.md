# GPU Audio Benchmark Suite

This is a placeholder repository while code is tidied and copied in.

DAFx Reviewers Update 2024-04-01 - code forthcoming ASAP; apologies for the delay. During development and testing, things were bunched up into a small number of files and I'm pulling these apart and commenting for readability.

Subdirectories:

- `cuda` for CUDA implementation. Tested on GTX 1080Ti and 4070, with most effort on the latter.
- `metal` for Objective-C Metal implementation. 
- `util` for any formatting and processing utility code as needed

All three directories are independent.

## Test platforms:

CUDA platform was tested on GTX 1080Ti and RTX 4070 cards, with development and most test coverage on the 4070.

Metal was tested on Apple Silicon and Intel MacOS platforms, with development and most test coverage on Apple Silicon.
