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

#endif /* globals_h */
