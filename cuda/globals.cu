#include "globals.cuh"

// Storage for globals.
int FS = 48000;

//int NTRACKS = 4 * 4096;
//int BUFSIZE = 512;
int NRUNS = 100;

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
