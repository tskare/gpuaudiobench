#ifndef globals_h
#define globals_h


/** Commandline arguments. **/
// CLEANUP: consider moving these to NSUserDefaults

// Audio buffer size in samples. Per-track.
// If changing this, please change BUFFERSIZE in the kernel.
extern int flagBufferSize;

extern NSString *benchmark;
extern int fs;
extern int nTracks;

extern bool dawsim;

// Cleanup: Move to BenchmarkFilters and pass into the kernel.
#define CONV1D_IRSIZE 256

#endif /* globals_h */
