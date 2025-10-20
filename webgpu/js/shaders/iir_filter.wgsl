// IIR Filter benchmark shader
// Implements a 2nd order biquad digital filter (Direct Form II)
// Common DSP operation used in audio processing

@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(2) var<uniform> coefficients: IIRCoefficients;
@group(0) @binding(3) var<storage, read_write> filter_states: array<f32>; // 2 states per track
@group(0) @binding(4) var<uniform> params: IIRParams;

struct IIRCoefficients {
    b0: f32,  // Feedforward coefficient 0
    b1: f32,  // Feedforward coefficient 1
    b2: f32,  // Feedforward coefficient 2
    a1: f32,  // Feedback coefficient 1
    a2: f32,  // Feedback coefficient 2
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

    // Check bounds
    if (track_id >= params.track_count) {
        return;
    }

    // Get filter state for this track (2 state variables per track)
    let state_index = track_id * 2u;
    var z1 = filter_states[state_index];
    var z2 = filter_states[state_index + 1u];

    // Process samples for this track
    let start_idx = track_id * params.buffer_size;

    for (var i = 0u; i < params.buffer_size; i++) {
        let sample_idx = start_idx + i;
        let x = input_buffer[sample_idx];

        // Direct Form II biquad implementation
        // y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
        // w[n] = x[n] - a1*w[n-1] - a2*w[n-2]

        let w = x - coefficients.a1 * z1 - coefficients.a2 * z2;
        let y = coefficients.b0 * w + coefficients.b1 * z1 + coefficients.b2 * z2;

        // Update state variables
        z2 = z1;
        z1 = w;

        // Store output
        output_buffer[sample_idx] = y;
    }

    // Save filter state for next iteration
    filter_states[state_index] = z1;
    filter_states[state_index + 1u] = z2;
}