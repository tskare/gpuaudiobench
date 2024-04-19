# GPU Audio Benchmark Suite

Status: CUDA code is being rebased and copied in.

DAFx Reviewers Update - CUDA code forthcoming ASAP; apologies for the delay. As the code is rebased onto XCode and CUDA Visual Studio example projects, I wanted to make sure the licenses are compatible. This has been verified for the Metal example. A LICENSE has now been added for GPL v3. 

Please note any LICENSEs in `cuda` or `metal` if spinning off a project; these are from the IDE sample projects used as a base for increased compatibility from the start.

Subdirectories:

- `cuda` for CUDA implementation. Tested on GTX 1080Ti and 4070, with most effort on the latter.
- `metal` for Objective-C Metal implementation. 
- `util` for any formatting and processing utility code as needed

All three directories are independent.

## Test platforms:

CUDA platform was tested on Windows, GTX 1080Ti and RTX 4070 cards, with development and most test coverage on the 4070. Open for bug reports on Windows or Linux. Will add CI for these.

Metal was tested on Apple Silicon and Intel MacOS platforms, with development and most test coverage on Apple Silicon. Going forward, development will use Apple Silicon and its integrated GPU though I'm open to bug reports for Intel+Metal.
