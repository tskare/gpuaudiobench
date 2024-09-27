#pragma once

// Revised base class for benchmarks
// Under constrution:
//   - The development branch diverged from public repo;
//     merge was a mess and I'd found some better ways
//     of structuring things in the meantime.
//   - Please hold off on extending using the old #ifdef
//     structure; the new version will be substantially
//     simpler and more flexible.

#include <string>
#include <vector>
#include <iostream>

#include "globals.cuh"

// Create a class GPUABenchmark that contains four cuda arrays, and a static construtor that can initialize them to a given parameter size.
// The class should also contain a virtual function that can be called to run the benchmark on the GPU.
class GPUABenchmark {
public:
	GPUABenchmark() = default;
	~GPUABenchmark() = default;

	// Empty interface; subclasses can specialize.
	class RunData {};
	enum class ValidationStatus {
		SUCCESS = 0,
		FAILURE = 1,
		FATAL = -1
	};
	class ValidationData {
		ValidationStatus status;
		std::vector<std::string> messages; // class can populate these for later logging.
	};
	// Run the benchmark.
	// RunData can be specialized in subclasses.
	virtual void run(RunData& runData) = 0;

	// Verify the benchmark
	virtual void validate(ValidationData* validationData /* [outparam] */) = 0;

	void InitDefaultBuffers(
		std::vector<int> host_input_buffer_sizes_bytes,
		std::vector<int> host_output_buffer_sizes_bytes,
		std::vector<int> device_input_buffer_sizes_bytes,
		std::vector<int> device_output_buffer_sizes_bytes);

	protected:
		// Default buffers if subclasses want shorthand.
		// TODO: support multiple buffers.
		void* dIn;
		void* dOut;
		void* hIn;
		void* hOut;
};
