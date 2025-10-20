//
// kernels_fdtd3d.metal
// MetalSwiftBench
//
// 3D Finite Difference Time Domain kernels for room acoustics simulation
//

#include <metal_stdlib>

using namespace metal;

/*
    FDTD3D solver overview
    ----------------------
    - Staggered grid: pressure lives at cell centers while velocity components live on the
      corresponding faces. Each axis allocates an extra layer of ghost cells so boundary damping
      can be applied without branching inside the inner stencil.
    - Absorbing boundary: Swift populates `absorptionCoeff` and pads the room dimensions. Inside the
      kernels we treat any ghost cell as part of a first-order absorbing layer by scaling pressure.
    - Stability: Swift computes `timeStep` from a CFL number < 1/sqrt(3), and we reuse the same
      constants (ρ₀ = 1.225 kg/m³) here so GPU and CPU references remain numerically aligned.
    - Source / receiver: both operate in the padded grid. A soft source (additive) with gain 0.1
      reduces the chance of numerical blow-up when the excitation is an impulse.
*/

// FDTD3D parameters structure matching Swift implementation
typedef struct {
    uint32_t nx, ny, nz;
    float soundSpeed;
    float spatialStep;
    float timeStep;
    float absorptionCoeff;
    uint32_t sourceX, sourceY, sourceZ;
    uint32_t receiverX, receiverY, receiverZ;
    uint32_t bufferSize;
    uint32_t trackCount;
} FDTD3DParams;

// Inline helper functions for 3D indexing
inline uint grid_index_3d(uint x, uint y, uint z, uint nx, uint ny) {
    return z * nx * ny + y * nx + x;
}

inline uint velocity_x_index(uint x, uint y, uint z, uint nx_plus_1, uint ny) {
    return z * nx_plus_1 * ny + y * nx_plus_1 + x;
}

inline uint velocity_y_index(uint x, uint y, uint z, uint nx, uint ny_plus_1) {
    return z * nx * ny_plus_1 + y * nx + x;
}

inline uint velocity_z_index(uint x, uint y, uint z, uint nx, uint ny) {
    return z * nx * ny + y * nx + x;
}

// Check if position is at boundary
inline bool is_boundary(uint x, uint y, uint z, uint nx, uint ny, uint nz) {
    return (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1);
}

//==============================================================================
// FDTD3D Velocity Update Kernel
//==============================================================================

kernel void fdtd3d_update_velocity(device const float* pressure           [[buffer(0)]],
                                   device float* velocity_x             [[buffer(1)]],
                                   device float* velocity_y             [[buffer(2)]],
                                   device float* velocity_z             [[buffer(3)]],
                                   constant FDTD3DParams& params        [[buffer(4)]],
                                   uint3 thread_position_in_grid        [[thread_position_in_grid]]) {
    
    const uint nx = params.nx;
    const uint ny = params.ny;
    const uint nz = params.nz;
    
    const float dt_over_rho_dx = params.timeStep / (1.225f * params.spatialStep);  // ρ₀ = 1.225 kg/m³
    
    const uint x = thread_position_in_grid.x;
    const uint y = thread_position_in_grid.y;
    const uint z = thread_position_in_grid.z;
    
    // Update x-directed face velocities; skip ghost faces so boundary damping stays outside.
    if (x < nx + 1 && y < ny && z < nz) {
        if (x > 0 && x < nx) {  // Interior points only
            const uint vx_idx = velocity_x_index(x, y, z, nx + 1, ny);
            const uint p_left = grid_index_3d(x - 1, y, z, nx, ny);
            const uint p_right = grid_index_3d(x, y, z, nx, ny);
            
            velocity_x[vx_idx] -= dt_over_rho_dx * (pressure[p_right] - pressure[p_left]);
        }
    }
    
    // Update y-directed faces on the staggered grid.
    if (x < nx && y < ny + 1 && z < nz) {
        if (y > 0 && y < ny) {  // Interior points only
            const uint vy_idx = velocity_y_index(x, y, z, nx, ny + 1);
            const uint p_front = grid_index_3d(x, y - 1, z, nx, ny);
            const uint p_back = grid_index_3d(x, y, z, nx, ny);
            
            velocity_y[vy_idx] -= dt_over_rho_dx * (pressure[p_back] - pressure[p_front]);
        }
    }
    
    // Update z-directed faces; guard against ghost layers on both ends.
    if (x < nx && y < ny && z < nz + 1) {
        if (z > 0 && z < nz) {  // Interior points only
            const uint vz_idx = velocity_z_index(x, y, z, nx, ny);
            const uint p_bottom = grid_index_3d(x, y, z - 1, nx, ny);
            const uint p_top = grid_index_3d(x, y, z, nx, ny);
            
            velocity_z[vz_idx] -= dt_over_rho_dx * (pressure[p_top] - pressure[p_bottom]);
        }
    }
}

//==============================================================================
// FDTD3D Pressure Update Kernel
//==============================================================================

kernel void fdtd3d_update_pressure(device float* pressure              [[buffer(0)]],
                                   device const float* velocity_x      [[buffer(1)]],
                                   device const float* velocity_y      [[buffer(2)]],
                                   device const float* velocity_z      [[buffer(3)]],
                                   constant FDTD3DParams& params       [[buffer(4)]],
                                   uint3 thread_position_in_grid       [[thread_position_in_grid]]) {
    
    const uint nx = params.nx;
    const uint ny = params.ny;
    const uint nz = params.nz;
    
    const float rho_c2_dt_over_dx = 1.225f * params.soundSpeed * params.soundSpeed * 
                                    params.timeStep / params.spatialStep;
    
    const uint x = thread_position_in_grid.x;
    const uint y = thread_position_in_grid.y;
    const uint z = thread_position_in_grid.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    const uint p_idx = grid_index_3d(x, y, z, nx, ny);
    
    // Interior cells: compute pressure from the divergence of the staggered velocity field.
    if (x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1) {
        const uint vx_left = velocity_x_index(x, y, z, nx + 1, ny);
        const uint vx_right = velocity_x_index(x + 1, y, z, nx + 1, ny);
        
        const uint vy_front = velocity_y_index(x, y, z, nx, ny + 1);
        const uint vy_back = velocity_y_index(x, y + 1, z, nx, ny + 1);
        
        const uint vz_bottom = velocity_z_index(x, y, z, nx, ny);
        const uint vz_top = velocity_z_index(x, y, z + 1, nx, ny);
        
        const float velocity_divergence = 
            (velocity_x[vx_right] - velocity_x[vx_left]) +
            (velocity_y[vy_back] - velocity_y[vy_front]) +
            (velocity_z[vz_top] - velocity_z[vz_bottom]);
        
        pressure[p_idx] -= rho_c2_dt_over_dx * velocity_divergence;
    }
    // Ghost layer: apply first-order absorption instead of reflecting the wavefront.
    else if (is_boundary(x, y, z, nx, ny, nz)) {
        // Treat absorptionCoeff as a linear reflection scale (1.0 absorbs everything).
        pressure[p_idx] *= (1.0f - params.absorptionCoeff);
    }
}

//==============================================================================
// FDTD3D Source Injection Kernel
//==============================================================================

kernel void fdtd3d_inject_source(device float* pressure              [[buffer(0)]],
                                 device const float* input_audio     [[buffer(1)]],
                                 constant FDTD3DParams& params       [[buffer(2)]],
                                 constant uint32_t& sample_index     [[buffer(3)]],
                                 uint thread_position_in_grid        [[thread_position_in_grid]]) {
    
    const uint track_idx = thread_position_in_grid;
    
    if (track_idx >= params.trackCount) return;
    
    const uint nx = params.nx;
    const uint ny = params.ny;
    
    // Get source position for this track (can be made track-dependent later)
    const uint source_x = params.sourceX;
    const uint source_y = params.sourceY;
    const uint source_z = params.sourceZ;
    
    // Locate the source cell inside the padded grid.
    const uint p_idx = grid_index_3d(source_x, source_y, source_z, nx, ny);
    
    // Get input sample for this track and sample index
    const uint audio_idx = track_idx * params.bufferSize + sample_index;
    const float input_sample = input_audio[audio_idx];
    
    // Inject source (soft source: add rather than overwrite)
    pressure[p_idx] += input_sample * 0.1f;  // Scale factor for stability
}

//==============================================================================
// FDTD3D Output Extraction Kernel
//==============================================================================

kernel void fdtd3d_extract_output(device const float* pressure        [[buffer(0)]],
                                  device float* output_audio          [[buffer(1)]],
                                  constant FDTD3DParams& params       [[buffer(2)]],
                                  constant uint32_t& sample_index     [[buffer(3)]],
                                  uint thread_position_in_grid        [[thread_position_in_grid]]) {
    
    const uint track_idx = thread_position_in_grid;
    
    if (track_idx >= params.trackCount) return;
    
    const uint nx = params.nx;
    const uint ny = params.ny;
    
    // Get receiver position for this track (can be made track-dependent later)
    const uint receiver_x = params.receiverX;
    const uint receiver_y = params.receiverY;
    const uint receiver_z = params.receiverZ;
    
    // Locate the receiver tap inside the padded grid.
    const uint p_idx = grid_index_3d(receiver_x, receiver_y, receiver_z, nx, ny);
    
    // Extract output sample for this track and sample index
    const uint audio_idx = track_idx * params.bufferSize + sample_index;
    output_audio[audio_idx] = pressure[p_idx] * 0.1f;  // Scale factor to normalize output
}

//==============================================================================
// FDTD3D Optimized Combined Update Kernel (Alternative Implementation)
//==============================================================================

// This kernel combines velocity and pressure updates for better performance
// by reducing memory bandwidth requirements and improving arithmetic intensity
kernel void fdtd3d_update_combined(device float* pressure             [[buffer(0)]],
                                   device float* velocity_x           [[buffer(1)]],
                                   device float* velocity_y           [[buffer(2)]],
                                   device float* velocity_z           [[buffer(3)]],
                                   constant FDTD3DParams& params      [[buffer(4)]],
                                   uint3 thread_position_in_grid      [[thread_position_in_grid]],
                                   threadgroup float* shared_pressure [[threadgroup(0)]]) {
    
    const uint nx = params.nx;
    const uint ny = params.ny;
    const uint nz = params.nz;
    
    const uint x = thread_position_in_grid.x;
    const uint y = thread_position_in_grid.y;
    const uint z = thread_position_in_grid.z;
    
    const uint local_x = x % 8;  // Assuming 8x8x4 threadgroup
    const uint local_y = y % 8;
    const uint local_z = z % 4;
    
    // Load pressure values into threadgroup memory for reuse
    if (x < nx && y < ny && z < nz) {
        const uint p_idx = grid_index_3d(x, y, z, nx, ny);
        const uint local_idx = local_z * 64 + local_y * 8 + local_x;  // 8*8*4 = 256
        shared_pressure[local_idx] = pressure[p_idx];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Update velocities using shared pressure data when possible
    // (Implementation would continue with optimized memory access patterns)
    // This is a framework - full implementation would be more complex
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Update pressures using updated velocities
    // (Similar pattern with optimized shared memory usage)
}

//==============================================================================
// FDTD3D Energy Calculation Kernel (for verification)
//==============================================================================

kernel void fdtd3d_calculate_energy(device const float* pressure       [[buffer(0)]],
                                    device const float* velocity_x     [[buffer(1)]],
                                    device const float* velocity_y     [[buffer(2)]],
                                    device const float* velocity_z     [[buffer(3)]],
                                    device float* energy_output        [[buffer(4)]],
                                    constant FDTD3DParams& params      [[buffer(5)]],
                                    uint3 thread_position_in_grid      [[thread_position_in_grid]]) {
    
    const uint nx = params.nx;
    const uint ny = params.ny;
    const uint nz = params.nz;
    
    const uint x = thread_position_in_grid.x;
    const uint y = thread_position_in_grid.y;
    const uint z = thread_position_in_grid.z;
    
    if (x >= nx || y >= ny || z >= nz) return;
    
    const uint p_idx = grid_index_3d(x, y, z, nx, ny);
    
    // Kinetic energy density: 0.5 * ρ * v² across the staggered velocity grid.
    float kinetic_energy = 0.0f;
    
    if (x < nx - 1) {
        const uint vx_idx = velocity_x_index(x + 1, y, z, nx + 1, ny);
        kinetic_energy += velocity_x[vx_idx] * velocity_x[vx_idx];
    }
    
    if (y < ny - 1) {
        const uint vy_idx = velocity_y_index(x, y + 1, z, nx, ny + 1);
        kinetic_energy += velocity_y[vy_idx] * velocity_y[vy_idx];
    }
    
    if (z < nz - 1) {
        const uint vz_idx = velocity_z_index(x, y, z + 1, nx, ny);
        kinetic_energy += velocity_z[vz_idx] * velocity_z[vz_idx];
    }
    
    kinetic_energy *= 0.5f * 1.225f;  // 0.5 * ρ₀
    
    // Potential energy density: 0.5 * p² / (ρ * c²).
    const float potential_energy = 0.5f * pressure[p_idx] * pressure[p_idx] / 
                                   (1.225f * params.soundSpeed * params.soundSpeed);
    
    // Store total energy for this grid point
    energy_output[p_idx] = kinetic_energy + potential_energy;
}
