@group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<f32>;
@group(0) @binding(2) var<uniform> params: DataTransferParams;

struct DataTransferParams {
    input_size: u32,
    output_size: u32,
    input_ratio: f32,
    output_ratio: f32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let total_threads = 64u;
    let input_elements_per_thread = (params.input_size + total_threads - 1u) / total_threads;
    let output_elements_per_thread = (params.output_size + total_threads - 1u) / total_threads;

    let input_start = thread_id * input_elements_per_thread;
    let input_end = min(input_start + input_elements_per_thread, params.input_size);

    let output_start = thread_id * output_elements_per_thread;
    let output_end = min(output_start + output_elements_per_thread, params.output_size);

    var sum: f32 = 0.0;
    for (var i = input_start; i < input_end; i++) {
        sum += input_buffer[i];
    }

    let output_value = sum * 0.000001; // Very small multiplier to avoid overflow

    for (var i = output_start; i < output_end; i++) {
        output_buffer[i] = output_value;
    }
}
