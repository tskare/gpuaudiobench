const MODE_PARAM_COUNT : u32 = 8u;
const PI : f32 = 3.14159265358979323846;

struct ModalParams {
    buffer_size : u32,
    num_modes : u32,
    output_tracks : u32,
    padding : u32,
};

@group(0) @binding(0) var<storage, read> mode_data : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer : array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params : ModalParams;

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
    let mode_index = gid.x;
    if (mode_index >= params.num_modes) {
        return;
    }

    let base = mode_index * MODE_PARAM_COUNT;
    let amplitude = mode_data[base + 0u];
    let frequency = mode_data[base + 1u];
    var state_re = mode_data[base + 3u];
    var state_im = mode_data[base + 4u];

    let output_track = mode_index % params.output_tracks;
    let cos_val = cos(2.0 * PI * frequency);
    let sin_val = sin(2.0 * PI * frequency);

    for (var sample : u32 = 0u; sample < params.buffer_size; sample += 1u) {
        let new_re = state_re * cos_val - state_im * sin_val;
        let new_im = state_re * sin_val + state_im * cos_val;
        state_re = new_re;
        state_im = new_im;

        let contribution = amplitude * state_re;
        let out_index = output_track * params.buffer_size + sample;
        atomic_add_float(&output_buffer[out_index], contribution);
    }
}
