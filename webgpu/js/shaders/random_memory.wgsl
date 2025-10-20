// Random Memory Read benchmark shader
// Simulates granular synthesis with non-sequential memory access patterns
// Tests memory bandwidth with poor cache locality

@group(0) @binding(0) var<storage, read> sample_memory: array<f32>; // Large sample buffer
@group(0) @binding(1) var<storage, read> playheads: array<u32>;     // Starting positions per track
@group(0) @binding(2) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(3) var<uniform> params: RandomMemoryParams;

struct RandomMemoryParams {
    buffer_size: u32,        // Samples per track (e.g., 512)
    track_count: u32,        // Number of parallel tracks
    sample_memory_size: u32, // Total samples in memory buffer
    _padding: u32,           // 16-byte alignment
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let track_id = global_id.x;

    // Check bounds
    if (track_id >= params.track_count) {
        return;
    }

    // Get starting playhead position for this track
    let playhead_start = playheads[track_id];

    // Calculate output buffer offset for this track
    let output_start = track_id * params.buffer_size;

    // Process samples for this track
    for (var i = 0u; i < params.buffer_size; i++) {
        // Calculate sample position (simulates granular synthesis)
        // Each track reads from a different starting position + offset
        let sample_position = playhead_start + i;

        // Bounds check for sample memory
        if (sample_position < params.sample_memory_size) {
            // Random access read - this stresses memory bandwidth
            // and tests cache performance with poor locality
            output_buffer[output_start + i] = sample_memory[sample_position];
        } else {
            // If we exceed bounds, output silence
            output_buffer[output_start + i] = 0.0;
        }
    }
}