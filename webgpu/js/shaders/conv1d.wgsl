struct Conv1DParams {
    buffer_size : u32,
    ir_length : u32,
    track_count : u32,
    use_constant_memory : u32,
};

@group(0) @binding(0) var<storage, read> input_audio : array<f32>;
@group(0) @binding(1) var<storage, read> impulse_responses : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_audio : array<f32>;
@group(0) @binding(3) var<uniform> params : Conv1DParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let track_index = gid.x;
    if (track_index >= params.track_count) {
        return;
    }

    let buf_size = params.buffer_size;
    let ir_length = params.ir_length;
    let input_base = track_index * buf_size;
    let ir_base = track_index * ir_length;
    let track_count = params.track_count;

    for (var sample_idx : u32 = 0u; sample_idx < buf_size; sample_idx = sample_idx + 1u) {
        var acc : f32 = 0.0;
        for (var tap : u32 = 0u; tap < ir_length && tap <= sample_idx; tap += 1u) {
            let input_idx = input_base + sample_idx - tap;
            let ir_idx = ir_base + tap;
            acc = acc + input_audio[input_idx] * impulse_responses[ir_idx];
        }

        let output_idx = sample_idx * track_count + track_index;
        output_audio[output_idx] = acc;
    }
}
