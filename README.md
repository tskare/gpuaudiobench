# GPU Audio Benchmark Suite

## Welcome DAFx24 readers

I enjoyed meeting and talking with you in September!

The conference poster is in the `media` subfolder. It is an HTML file that may be loaded in your browser.

## Recent and Upcoming changes

Linux: Makefile change to support systems that didn't have `cuda-samples` headers set system-wide. Documentation added to the `cuda` subdir.

Note that CUDA and `gcc` have compiler version pairing constraints.

## Upcoming changes

A significant merge of a development branch is incoming with:

+ CUDA: revised top-level class rather than `#ifdef`s.
+ Facility for specifying multiple benchmarks, for a smoother demo (you may currently rerun the executable without recompiling to accomplish this).
+ additional client-side validations that were omitted in this repo.
+ fixes made for the development/paper version that need to be merged to this repo.
+ an additional benchmark

You may access the earlier version with the `v1.0` tag.

## Project Overview

This project contains microbenchmarks and domain-specific benchmarks towards qualifying platforms for real-time GPGPU audio use. 

Subdirectories:

- `cuda` for CUDA implementation.
- `metal` for Objective-C Metal implementation. 
- `media` for artifacts like the DAFx24 poster. 

Directories are independent but checked out in the same repository. Each directory may contain its own `README.md` with details for that platform.

This project is GPL-3 licensed but please mail if this does not work for you. Please note any LICENSEs in `cuda` or `metal` speficially if spinning off a project; these are from the IDE sample projects used as a base for increased compatibility on new developers' machines where
the relevant GPGPU toolkits are likely already installed.

We aim to require no dependencies beyond the relevant GPGPU toolkit and the C++ standard library. We aim to require minimal project configuration.

## Status

Various cleanup tasks are in progress, usually indicated by CLEANUP: or TODO: in the code.

The code is expected to build on MacOS and Windows, with Linux supported but to be tested. In case of issues, please file an issue on GitHub or reach out to travisskare@gmail.com.

## Test platforms:

CUDA platform was tested on Windows, GTX 1080Ti and RTX 4070 cards, with development and most test coverage on the 4070. Please report bugs via GitHub, CCRMA GitLab, or email. I'll aim to add binary releases on GitHub for these.

Code was tested on Linux at an earlier point, but not the current version (I reimaged that machine to Windows). `cuda/Makefile` contains the Linux Makefile.

Metal was tested on Apple Silicon and Intel MacOS platforms, with development and most test coverage on Apple Silicon. Going forward, development will use Apple Silicon and its integrated GPU though I'm open to bug reports for Intel+Metal regressions.
