const RHO0 : f32 = 1.225;
const PI : f32 = 3.14159265358979323846;

struct FDTD3DParams {
    nx : u32,
    ny : u32,
    nz : u32,
    track_count : u32,
    sound_speed : f32,
    spatial_step : f32,
    time_step : f32,
    absorption : f32,
    source_x : u32,
    source_y : u32,
    source_z : u32,
    buffer_size : u32,
};

struct SampleIndexUniform {
    value : u32,
    padding0 : u32,
    padding1 : u32,
    padding2 : u32,
};

fn grid_index_3d(x : u32, y : u32, z : u32, nx : u32, ny : u32) -> u32 {
    return z * nx * ny + y * nx + x;
}

fn velocity_x_index(x : u32, y : u32, z : u32, nx_plus_1 : u32, ny : u32) -> u32 {
    return z * nx_plus_1 * ny + y * nx_plus_1 + x;
}

fn velocity_y_index(x : u32, y : u32, z : u32, nx : u32, ny_plus_1 : u32) -> u32 {
    return z * nx * ny_plus_1 + y * nx + x;
}

fn velocity_z_index(x : u32, y : u32, z : u32, nx : u32, ny : u32) -> u32 {
    return z * nx * ny + y * nx + x;
}

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

@group(0) @binding(0) var<storage, read> pressure_for_velocity : array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_x : array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_y : array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_z : array<f32>;
@group(0) @binding(4) var<uniform> params_velocity : FDTD3DParams;

@compute @workgroup_size(4, 4, 4)
fn update_velocity(@builtin(global_invocation_id) gid : vec3<u32>) {
    let nx = params_velocity.nx;
    let ny = params_velocity.ny;
    let nz = params_velocity.nz;

    if (gid.x < nx + 1u && gid.y < ny && gid.z < nz) {
        if (gid.x > 0u && gid.x < nx) {
            let vx_idx = velocity_x_index(gid.x, gid.y, gid.z, nx + 1u, ny);
            let p_right = pressure_for_velocity[grid_index_3d(gid.x, gid.y, gid.z, nx, ny)];
            let p_left = pressure_for_velocity[grid_index_3d(gid.x - 1u, gid.y, gid.z, nx, ny)];
            let dt_over_rho_dx = params_velocity.time_step / (RHO0 * params_velocity.spatial_step);
            velocity_x[vx_idx] = velocity_x[vx_idx] - dt_over_rho_dx * (p_right - p_left);
        }
    }

    if (gid.x < nx && gid.y < ny + 1u && gid.z < nz) {
        if (gid.y > 0u && gid.y < ny) {
            let vy_idx = velocity_y_index(gid.x, gid.y, gid.z, nx, ny + 1u);
            let p_back = pressure_for_velocity[grid_index_3d(gid.x, gid.y, gid.z, nx, ny)];
            let p_front = pressure_for_velocity[grid_index_3d(gid.x, gid.y - 1u, gid.z, nx, ny)];
            let dt_over_rho_dx = params_velocity.time_step / (RHO0 * params_velocity.spatial_step);
            velocity_y[vy_idx] = velocity_y[vy_idx] - dt_over_rho_dx * (p_back - p_front);
        }
    }

    if (gid.x < nx && gid.y < ny && gid.z < nz + 1u) {
        if (gid.z > 0u && gid.z < nz) {
            let vz_idx = velocity_z_index(gid.x, gid.y, gid.z, nx, ny);
            let p_top = pressure_for_velocity[grid_index_3d(gid.x, gid.y, gid.z, nx, ny)];
            let p_bottom = pressure_for_velocity[grid_index_3d(gid.x, gid.y, gid.z - 1u, nx, ny)];
            let dt_over_rho_dx = params_velocity.time_step / (RHO0 * params_velocity.spatial_step);
            velocity_z[vz_idx] = velocity_z[vz_idx] - dt_over_rho_dx * (p_top - p_bottom);
        }
    }
}

@group(0) @binding(5) var<storage, read_write> pressure_field : array<f32>;
@group(0) @binding(6) var<storage, read> velocity_x_for_pressure : array<f32>;
@group(0) @binding(7) var<storage, read> velocity_y_for_pressure : array<f32>;
@group(0) @binding(8) var<storage, read> velocity_z_for_pressure : array<f32>;
@group(0) @binding(9) var<uniform> params_pressure : FDTD3DParams;

@compute @workgroup_size(4, 4, 4)
fn update_pressure(@builtin(global_invocation_id) gid : vec3<u32>) {
    let nx = params_pressure.nx;
    let ny = params_pressure.ny;
    let nz = params_pressure.nz;

    if (gid.x >= nx || gid.y >= ny || gid.z >= nz) {
        return;
    }

    if (gid.x > 0u && gid.x < nx - 1u && gid.y > 0u && gid.y < ny - 1u && gid.z > 0u && gid.z < nz - 1u) {
        let idx = grid_index_3d(gid.x, gid.y, gid.z, nx, ny);
        let vx_right = velocity_x_for_pressure[velocity_x_index(gid.x + 1u, gid.y, gid.z, nx + 1u, ny)];
        let vx_left = velocity_x_for_pressure[velocity_x_index(gid.x, gid.y, gid.z, nx + 1u, ny)];
        let vy_back = velocity_y_for_pressure[velocity_y_index(gid.x, gid.y + 1u, gid.z, nx, ny + 1u)];
        let vy_front = velocity_y_for_pressure[velocity_y_index(gid.x, gid.y, gid.z, nx, ny + 1u)];
        let vz_top = velocity_z_for_pressure[velocity_z_index(gid.x, gid.y, gid.z + 1u, nx, ny)];
        let vz_bottom = velocity_z_for_pressure[velocity_z_index(gid.x, gid.y, gid.z, nx, ny)];

        let divergence = (vx_right - vx_left) + (vy_back - vy_front) + (vz_top - vz_bottom);
        let coeff = RHO0 * params_pressure.sound_speed * params_pressure.sound_speed * params_pressure.time_step / params_pressure.spatial_step;
        pressure_field[idx] = pressure_field[idx] - coeff * divergence;
    } else {
        // simple absorbing boundary
        let idx = grid_index_3d(gid.x, gid.y, gid.z, nx, ny);
        pressure_field[idx] = pressure_field[idx] * (1.0 - params_pressure.absorption);
    }
}

@group(0) @binding(10) var<storage, read_write> pressure_inject : array<f32>;
@group(0) @binding(11) var<storage, read> input_audio : array<f32>;
@group(0) @binding(12) var<uniform> params_inject : FDTD3DParams;
@group(0) @binding(13) var<uniform> sample_index_inject : SampleIndexUniform;

@compute @workgroup_size(64)
fn inject_source(@builtin(global_invocation_id) gid : vec3<u32>) {
    let track_idx = gid.x;
    if (track_idx > 0u) {
        return; // inject using track 0 to avoid write conflicts
    }

    let sample_index = sample_index_inject.value;
    if (sample_index >= params_inject.buffer_size) {
        return;
    }

    let source_idx = grid_index_3d(params_inject.source_x, params_inject.source_y, params_inject.source_z, params_inject.nx, params_inject.ny);
    let sample = input_audio[sample_index];
    pressure_inject[source_idx] = pressure_inject[source_idx] + sample * 0.1;
}

@group(0) @binding(14) var<storage, read> pressure_extract : array<f32>;
@group(0) @binding(15) var<storage, read_write> output_audio : array<f32>;
@group(0) @binding(16) var<uniform> params_extract : FDTD3DParams;
@group(0) @binding(17) var<uniform> sample_index_extract : SampleIndexUniform;
@group(0) @binding(18) var<storage, read> receiver_positions : array<vec4<u32>>;

@compute @workgroup_size(64)
fn extract_output(@builtin(global_invocation_id) gid : vec3<u32>) {
    let track_idx = gid.x;
    if (track_idx >= params_extract.track_count) {
        return;
    }

    let sample_index = sample_index_extract.value;
    if (sample_index >= params_extract.buffer_size) {
        return;
    }

    let pos = receiver_positions[track_idx];
    let idx = grid_index_3d(pos.x, pos.y, pos.z, params_extract.nx, params_extract.ny);
    let output_idx = track_idx * params_extract.buffer_size + sample_index;
    output_audio[output_idx] = pressure_extract[idx];
}
