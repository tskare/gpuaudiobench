struct GainStatsParams {
    buffer_size : u32,
    track_count : u32,
    gain_value : f32,
    _padding : f32,
};

@group(0) @binding(0) var<storage, read> input_audio : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_audio : array<f32>;
@group(0) @binding(2) var<storage, read_write> stats_buffer : array<f32>;
@group(0) @binding(3) var<uniform> params : GainStatsParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let track_index = gid.x;
    if (track_index >= params.track_count) {
        return;
    }

    var mean_acc : f32 = 0.0;
    var max_value : f32 = -3.402823e+38;
    let base = track_index * params.buffer_size;

    for (var sample_index : u32 = 0u; sample_index < params.buffer_size; sample_index += 1u) {
        let idx = base + sample_index;
        let sample = input_audio[idx];
        output_audio[idx] = sample * params.gain_value;
        mean_acc = mean_acc + sample;
        if (sample > max_value) {
            max_value = sample;
        }
    }

    let mean_value = mean_acc / f32(params.buffer_size);
    let stats_base = track_index * 2u;
    stats_buffer[stats_base] = mean_value;
    stats_buffer[stats_base + 1u] = max_value;
}
