@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(2) var<uniform> coefficients: IIRCoefficients;
@group(0) @binding(3) var<storage, read_write> filter_states: array<f32>; // 2 states per track
@group(0) @binding(4) var<uniform> params: IIRParams;

struct IIRCoefficients {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    _padding: array<f32, 3>, // Padding for 16-byte alignment
}

struct IIRParams {
    buffer_size: u32,
    track_count: u32,
    _padding: array<u32, 2>, // Padding for 16-byte alignment
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let track_id = global_id.x;

    if (track_id >= params.track_count) {
        return;
    }

    let state_index = track_id * 2u;
    var z1 = filter_states[state_index];
    var z2 = filter_states[state_index + 1u];

    let start_idx = track_id * params.buffer_size;

    for (var i = 0u; i < params.buffer_size; i++) {
        let sample_idx = start_idx + i;
        let x = input_buffer[sample_idx];

        // y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
        // w[n] = x[n] - a1*w[n-1] - a2*w[n-2]

        let w = x - coefficients.a1 * z1 - coefficients.a2 * z2;
        let y = coefficients.b0 * w + coefficients.b1 * z1 + coefficients.b2 * z2;

        z2 = z1;
        z1 = w;

        output_buffer[sample_idx] = y;
    }

    filter_states[state_index] = z1;
    filter_states[state_index + 1u] = z2;
}
