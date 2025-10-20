const MAX_BUFFER_SIZE : u32 = 1024u;

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

var<workgroup> shared_output : array<atomic<u32>, MAX_BUFFER_SIZE>;

fn atomic_add_float_workgroup(ptr : ptr<workgroup, atomic<u32>>, value : f32) {
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
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>
) {
    let buffer_size = params.buffer_size;
    let use_shared = buffer_size <= MAX_BUFFER_SIZE;

    if (use_shared) {
        for (var i : u32 = lid.x; i < buffer_size; i += 64u) {
            atomicStore(&shared_output[i], bitcast<u32>(0.0));
        }
    }
    workgroupBarrier();

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

    for (var sample : u32 = 0u; sample < buffer_size; sample += 1u) {

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
            let contribution = (forward_sample + backward_sample) * 0.5;
            if (use_shared) {
                atomic_add_float_workgroup(&shared_output[sample], contribution);
            } else {
                atomic_add_float(&output_buffer[sample], contribution);
            }
        }

    }

    workgroupBarrier();

    if (use_shared) {
        for (var idx : u32 = lid.x; idx < buffer_size; idx += 64u) {
            let accum = bitcast<f32>(atomicLoad(&shared_output[idx]));
            if (accum != 0.0) {
                atomic_add_float(&output_buffer[idx], accum);
            }
        }
    }
}
