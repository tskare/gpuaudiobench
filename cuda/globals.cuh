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
// Storage and defaults are in globals.cu.

// Commandline parameters:
extern int NTRACKS;
extern int FS;
extern int BUFSIZE;
extern int NRUNS;
extern std::string OUTPUT_FILE;
extern bool JSON_OUTPUT;

// DAW-Simultion configurable parameters
#define ENABLE_DAWSIM_SLEEP false
#define SLEEP_MS 90   // Inclusive of processing time
#define ENABLE_DAWSIM_SPIN false

constexpr int numElements = 50000;

// Shared functions
// TODO: Write to stdout or Windows console.
// Our driver script watches for this and copies to another file to generate tables.
#define OUTFILE "c:\\tmp\\latencies.txt"
void writeVectorToFile(const std::vector<float>& vec, const std::string& filename);
void printVectorStats(const std::vector<float>& vec);
void writeCSVResults(const std::vector<float>& vec, const std::string& benchmarkName, const std::string& filename);
void writeJSONResults(const std::vector<float>& vec, const std::string& benchmarkName, const std::string& filename = "");
std::string generateJSONResults(const std::vector<float>& vec, const std::string& benchmarkName);
