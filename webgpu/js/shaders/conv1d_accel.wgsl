const WORKGROUP_SIZE : u32 = 64u;

struct Conv1DParams {
    buffer_size : u32,
    ir_length : u32,
    track_count : u32,
    mode : u32,
};

@group(0) @binding(0) var<storage, read> input_audio : array<f32>;
@group(0) @binding(1) var<storage, read> impulse_responses : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_audio : array<f32>;
@group(0) @binding(3) var<uniform> params : Conv1DParams;

var<workgroup> shared_ir : array<f32, WORKGROUP_SIZE>;

fn compute_sample(
    track_idx : u32,
    sample_idx : u32,
    buffer_size : u32,
    ir_length : u32,
    local_lane : u32
) -> f32 {
    let input_base = track_idx * buffer_size;
    let ir_base = track_idx * ir_length;

    var accumulator : f32 = 0.0;

    for (var tap_base : u32 = 0u; tap_base < ir_length; tap_base += WORKGROUP_SIZE) {
        let tap_index = tap_base + local_lane;
        if (local_lane < WORKGROUP_SIZE && tap_index < ir_length) {
            shared_ir[local_lane] = impulse_responses[ir_base + tap_index];
        }
        workgroupBarrier();

        for (var tile_offset : u32 = 0u; tile_offset < WORKGROUP_SIZE; tile_offset += 1u) {
            let tap = tap_base + tile_offset;
            if (tap >= ir_length || tap > sample_idx) {
                break;
            }
            let input_index = input_base + (sample_idx - tap);
            accumulator = accumulator + shared_ir[tile_offset] * input_audio[input_index];
        }

        workgroupBarrier();
    }

    return accumulator;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
    @builtin(workgroup_id) wid : vec3<u32>
) {
    let buffer_size = params.buffer_size;
    let ir_length = params.ir_length;
    let track_idx = gid.y;
    if (track_idx >= params.track_count) {
        return;
    }

    let tile_base = wid.x * WORKGROUP_SIZE;
    let sample_idx = tile_base + lid.x;
    if (sample_idx >= buffer_size) {
        return;
    }

    let result = compute_sample(track_idx, sample_idx, buffer_size, ir_length, lid.x);
    let output_index = sample_idx * params.track_count + track_idx;
    output_audio[output_index] = result;
}
