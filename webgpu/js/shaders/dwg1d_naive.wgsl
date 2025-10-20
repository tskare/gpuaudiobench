const PI : f32 = 3.14159265358979323846;

struct WaveguideState {
    length : u32,
    input_tap_pos : u32,
    output_tap_pos : u32,
    write_pos : u32,
    gain : f32,
    reflection : f32,
    damping : f32,
    padding : f32,
};

struct DWGParams {
    buffer_size : u32,
    max_length : u32,
    num_waveguides : u32,
    padding : u32,
};

@group(0) @binding(0) var<storage, read> waveguides : array<WaveguideState>;
@group(0) @binding(1) var<storage, read_write> delay_forward : array<f32>;
@group(0) @binding(2) var<storage, read_write> delay_backward : array<f32>;
@group(0) @binding(3) var<storage, read> input_signal : array<f32>;
@group(0) @binding(4) var<storage, read_write> output_buffer : array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params : DWGParams;

fn atomic_add_float(ptr : ptr<storage, atomic<u32>>, value : f32) {
    var old_bits : u32 = atomicLoad(ptr);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_value = old_value + value;
        let new_bits = bitcast<u32>(new_value);
        let result = atomicCompareExchangeWeak(ptr, old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
        old_bits = result.old_value;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let wg_index = gid.x;
    if (wg_index >= params.num_waveguides) {
        return;
    }

    let state = waveguides[wg_index];
    let length = state.length;
    if (length == 0u) {
        return;
    }

    let base = wg_index * params.max_length;
    let half_length = length >> 1u;

    for (var sample : u32 = 0u; sample < params.buffer_size; sample += 1u) {

        let input_value = input_signal[sample] * state.gain;
        let write_index = (state.write_pos + sample) % length;
        let forward_index = base + write_index;
        let backward_index = base + ((state.write_pos + sample + half_length) % length);

        var forward_sample = delay_forward[forward_index] * state.damping;
        var backward_sample = delay_backward[backward_index] * state.damping;

        if (write_index == state.input_tap_pos) {
            forward_sample = forward_sample + input_value;
            backward_sample = backward_sample + input_value;
        }

        let new_forward = backward_sample * state.reflection + input_value;
        let new_backward = forward_sample * state.reflection + input_value;

        delay_forward[forward_index] = new_forward;
        delay_backward[backward_index] = new_backward;

        if (write_index == state.output_tap_pos) {
            let output_value = (forward_sample + backward_sample) * 0.5;
            atomic_add_float(&output_buffer[sample], output_value);
        }

    }
}
