@group(0) @binding(0) var<storage, read> sample_memory: array<f32>;
@group(0) @binding(1) var<storage, read> playheads: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(3) var<uniform> params: RandomMemoryParams;

struct RandomMemoryParams {
    buffer_size: u32,
    track_count: u32,
    sample_memory_size: u32,
    _padding: u32,           // 16-byte alignment
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let track_id = global_id.x;

    if (track_id >= params.track_count) {
        return;
    }

    let playhead_start = playheads[track_id];

    let output_start = track_id * params.buffer_size;

    for (var i = 0u; i < params.buffer_size; i++) {
        let sample_position = playhead_start + i;

        if (sample_position < params.sample_memory_size) {
            output_buffer[output_start + i] = sample_memory[sample_position];
        } else {
            output_buffer[output_start + i] = 0.0;
        }
    }
}
