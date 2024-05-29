#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

using std::cout;
using std::endl;
using std::vector;


// Global constants shared between various files.

// CLEANUP: These should be made commandline parameters.
#define NTRACKS 4*4096
#define FS 48000
#define BUFSIZE 512
#define NRUNS 100
// DAW-Simultion configurable parameters
#define ENABLE_DAWSIM_SLEEP false
#define SLEEP_MS 90   // Inclusive of processing time
#define ENABLE_DAWSIM_SPIN false

constexpr int numElements = 50000;

// Choose one of N benchmarks here.
// This will be converted to a commandline parameter.
#define DO_BENCHMARK_IO true
#define DO_BENCHMARK_GAINSTATS false
#define DO_BENCHMARK_MODAL false
#define DO_BENCHMARK_RNDMEM false

// Shared functions
// TODO: Write to stdout or Windows console.
// Our driver script watches for this and copies to another file to generate tables.
#define OUTFILE "c:\\tmp\\latencies.txt"
void writeVectorToFile(const std::vector<float>& vec, const std::string& filename);
void printVectorStats(const std::vector<float>& vec);
