#pragma once

// Benchmark declarations for IO tests.

// Configurable parameters
// CLEANUP: These should be commandline parameters

// I/O mix, [1..99], as a percent of total.
#define IOTEST_INFRAC 1
#define IOTEST_INSCALE (IOTEST_INFRAC / 100.0f)
#define IOTEST_OUTSCALE (1.0f - IOTEST_INSCALE)

constexpr int ONEMEG_OF_FLOATS = (1024 * 1024 / 4);
constexpr int IOTEST_BUFSIZE = (10 * ONEMEG_OF_FLOATS);
constexpr int IOTEST_INBUFCOUNT = (((int)(IOTEST_BUFSIZE * IOTEST_INSCALE)));
constexpr int IOTEST_OUTBUFCOUNT = (((int)(IOTEST_BUFSIZE * IOTEST_OUTSCALE)));

// Declarations
void SetupBenchmarkIO(float** h_inBuf, float** h_outBuf, float** d_inBuf, float** d_outBuf,
    const int IOTEST_INBUFCOUNT, const int IOTEST_OUTBUFCOUNT);