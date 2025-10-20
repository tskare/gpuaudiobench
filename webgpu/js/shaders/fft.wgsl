struct FFTParams {
    buffer_size : u32,
    fft_size : u32,
    output_size : u32,
    track_count : u32,
};

@group(0) @binding(0) var<storage, read> input_audio : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_fft : array<f32>;
@group(0) @binding(2) var<uniform> params : FFTParams;

const PI : f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let track_index = gid.x;
    if (track_index >= params.track_count) {
        return;
    }

    let input_base = track_index * params.buffer_size;
    let output_base = track_index * params.output_size * 2u;
    let sample_limit = min(params.buffer_size, params.fft_size);

    for (var bin : u32 = 0u; bin < params.output_size; bin += 1u) {

        var sum_real : f32 = 0.0;
        var sum_imag : f32 = 0.0;

        for (var n : u32 = 0u; n < sample_limit; n += 1u) {
            let angle = -2.0 * PI * f32(bin) * f32(n) / f32(params.fft_size);
            let sample = input_audio[input_base + n];
            sum_real = sum_real + sample * cos(angle);
            sum_imag = sum_imag + sample * sin(angle);
        }

        let base = output_base + bin * 2u;
        output_fft[base] = sum_real;
        output_fft[base + 1u] = sum_imag;
    }
}
