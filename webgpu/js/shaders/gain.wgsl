@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(2) var<uniform> params: GainParams;

struct GainParams {
    buffer_size: u32,
    track_count: u32,
    gain_value: f32,
    _padding: f32,  // Ensure 16-byte alignment
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let track_id = global_id.x;

    if (track_id >= params.track_count) {
        return;
    }

    let start_idx = track_id * params.buffer_size;
    let end_idx = start_idx + params.buffer_size;

    for (var i = start_idx; i < end_idx; i++) {
        output_buffer[i] = params.gain_value * input_buffer[i];
    }
}
