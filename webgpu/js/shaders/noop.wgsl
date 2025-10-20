// NoOp benchmark shader
// Measures GPU kernel launch overhead with minimal computation

@group(0) @binding(0) var<storage, read_write> dummy_buffer: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Minimal no-op computation to prevent shader optimization
    // We need to touch at least one buffer element to ensure the kernel actually runs
    let index = global_id.x;
    if (index == 0u) {
        dummy_buffer[0] = dummy_buffer[0] + 0.0;
    }
}