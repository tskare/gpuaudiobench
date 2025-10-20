#include "globals.cuh"

// Storage for globals.
int FS = 48000;
int NTRACKS = 256;  // Default value, can be overridden by --nTracks
int BUFSIZE = 512;  // Default value, can be overridden by --bufferSize
int NRUNS = 100;
std::string OUTPUT_FILE = "";  // Default is no output file
bool JSON_OUTPUT = false;  // Default is text output

//==============================================================================
// LEGACY UTILITIES - MARKED FOR DEPRECATION
// These functions rely on global state and should be migrated to bench_utils.cu
// with proper parameter passing. See P2 tasks in PLAN-cuda-consolidated.md
//==============================================================================

#define OUTFILE "c:\\tmp\\latencies.txt"
void writeVectorToFile(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& val : vec) {
        file << val << std::endl;
    }
    file.close();
}

void printVectorStats(const std::vector<float>& vec) {
    float sum = 0.0f;
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (const auto& val : vec) {
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    float avg = sum / vec.size();
    std::cout << "Min: " << min << " Max: " << max << " Avg: " << avg << std::endl;

    // Sort vector
    std::vector<float> sortedVec = vec;
    std::sort(sortedVec.begin(), sortedVec.end());
    // Print median
    float median = 0.0f;
    if (sortedVec.size() % 2 == 0) {
        median = (sortedVec[sortedVec.size() / 2 - 1] + sortedVec[sortedVec.size() / 2]) / 2;
    }
    else {
        median = sortedVec[sortedVec.size() / 2];
    }
    const float p50 = sortedVec[sortedVec.size() * 0.50];
    const float p95 = sortedVec[sortedVec.size() * 0.95];
    const float p99 = sortedVec[sortedVec.size() * 0.99];
    std::cout << "p50: " << p50 << " p95: " << p95 << " p99: " << p99 << std::endl;

	// Output human-readable summary
    const float latencyThreshold = 1000.0f * BUFSIZE / FS;
    std::cout << "Latency threshold (" << FS << "Hz" << "): " << latencyThreshold << " ms" << std::endl;
    if (p50 > latencyThreshold) {
		std::cout << "WARNING: p50 exceeds threshold" << std::endl;
    }
    else if (p95 > latencyThreshold) {
		std::cout << "WARNING: p95 exceeds threshold" << std::endl;
	} else if (p99 > latencyThreshold) {
		std::cout << "WARNING: p99 exceeds threshold" << std::endl;
	} else {
		std::cout << "OK: Measured latencies within threshold. Please consider a margin of safety." << std::endl;
    }
}

void writeCSVResults(const std::vector<float>& vec, const std::string& benchmarkName, const std::string& filename) {
    if (filename.empty()) return;

    // Calculate statistics
    float sum = 0.0f;
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (const auto& val : vec) {
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    float avg = sum / vec.size();

    std::vector<float> sortedVec = vec;
    std::sort(sortedVec.begin(), sortedVec.end());

    const float p50 = sortedVec[sortedVec.size() * 0.50];
    const float p95 = sortedVec[sortedVec.size() * 0.95];
    const float p99 = sortedVec[sortedVec.size() * 0.99];
    const float latencyThreshold = 1000.0f * BUFSIZE / FS;

    // Check if file exists to determine if we need to write header
    bool writeHeader = false;
    std::ifstream testFile(filename);
    if (!testFile.good()) {
        writeHeader = true;
    }
    testFile.close();

    std::ofstream file(filename, std::ios::app);
    if (writeHeader) {
        file << "benchmark,fs,bufferSize,nTracks,nRuns,min_ms,max_ms,avg_ms,p50_ms,p95_ms,p99_ms,threshold_ms,meets_deadline\n";
    }

    bool meetsDeadline = (p99 <= latencyThreshold);

    file << benchmarkName << ","
         << FS << ","
         << BUFSIZE << ","
         << NTRACKS << ","
         << vec.size() << ","
         << min << ","
         << max << ","
         << avg << ","
         << p50 << ","
         << p95 << ","
         << p99 << ","
         << latencyThreshold << ","
         << (meetsDeadline ? "true" : "false") << "\n";

    file.close();
    std::cout << "Results saved to: " << filename << std::endl;
}

void writeJSONResults(const std::vector<float>& vec, const std::string& benchmarkName, const std::string& filename) {
    if (filename.empty()) {
        // Output to stdout if no filename specified
        std::cout << generateJSONResults(vec, benchmarkName) << std::endl;
        return;
    }

    std::ofstream file(filename);
    file << generateJSONResults(vec, benchmarkName);
    file.close();
    std::cout << "JSON results saved to: " << filename << std::endl;
}

std::string generateJSONResults(const std::vector<float>& vec, const std::string& benchmarkName) {
    // Calculate statistics
    float sum = 0.0f;
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (const auto& val : vec) {
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
    }
    float avg = sum / vec.size();

    std::vector<float> sortedVec = vec;
    std::sort(sortedVec.begin(), sortedVec.end());

    const float p50 = sortedVec[sortedVec.size() * 0.50];
    const float p95 = sortedVec[sortedVec.size() * 0.95];
    const float p99 = sortedVec[sortedVec.size() * 0.99];
    const float latencyThreshold = 1000.0f * BUFSIZE / FS;
    const bool meetsDeadline = (p99 <= latencyThreshold);

    // Build JSON string manually (avoiding dependencies)
    std::string json = "{\n";
    json += "  \"benchmark\": \"" + benchmarkName + "\",\n";
    json += "  \"configuration\": {\n";
    json += "    \"fs\": " + std::to_string(FS) + ",\n";
    json += "    \"bufferSize\": " + std::to_string(BUFSIZE) + ",\n";
    json += "    \"nTracks\": " + std::to_string(NTRACKS) + ",\n";
    json += "    \"nRuns\": " + std::to_string((int)vec.size()) + "\n";
    json += "  },\n";
    json += "  \"statistics\": {\n";
    json += "    \"min_ms\": " + std::to_string(min) + ",\n";
    json += "    \"max_ms\": " + std::to_string(max) + ",\n";
    json += "    \"avg_ms\": " + std::to_string(avg) + ",\n";
    json += "    \"p50_ms\": " + std::to_string(p50) + ",\n";
    json += "    \"p95_ms\": " + std::to_string(p95) + ",\n";
    json += "    \"p99_ms\": " + std::to_string(p99) + "\n";
    json += "  },\n";
    json += "  \"deadline\": {\n";
    json += "    \"threshold_ms\": " + std::to_string(latencyThreshold) + ",\n";
    json += "    \"meets_deadline\": " + std::string(meetsDeadline ? "true" : "false") + "\n";
    json += "  }\n";
    json += "}\n";

    return json;
}
