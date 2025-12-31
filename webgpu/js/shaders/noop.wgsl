@group(0) @binding(0) var<storage, read_write> dummy_buffer: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Touch one element to avoid dead-code elimination.
    let index = global_id.x;
    if (index == 0u) {
        dummy_buffer[0] = dummy_buffer[0] + 0.0;
    }
}
