#pragma once

// 3D FDTD benchmark for room acoustics simulation.

#include "bench_base.cuh"

//==============================================================================
// Configuration Parameters
//==============================================================================

// Room dimensions (configurable)
constexpr int kFDTD3D_RoomX = 50;         // Room width in grid points
constexpr int kFDTD3D_RoomY = 50;         // Room height in grid points
constexpr int kFDTD3D_RoomZ = 50;         // Room depth in grid points

// Physical parameters
constexpr float kFDTD3D_SoundSpeed = 343.0f;       // Speed of sound (m/s)
constexpr float kFDTD3D_SpatialStep = 0.01f;       // Grid spacing (m)
constexpr float kFDTD3D_AirDensity = 1.225f;       // Air density (kg/m³)
constexpr float kFDTD3D_AbsorptionCoeff = 0.2f;    // Wall absorption coefficient

// Numerical parameters
constexpr float kFDTD3D_CFLNumber = 0.5f;          // CFL stability factor (< 1/√3)
constexpr int kFDTD3D_StepsPerSample = 3;          // FDTD steps per audio sample

// Source and receiver positions (grid coordinates)
constexpr int kFDTD3D_SourceX = 25;        // Source X position
constexpr int kFDTD3D_SourceY = 25;        // Source Y position
constexpr int kFDTD3D_SourceZ = 5;         // Source Z position (near floor)

constexpr int kFDTD3D_ReceiverX = 40;      // Receiver X position
constexpr int kFDTD3D_ReceiverY = 15;      // Receiver Y position
constexpr int kFDTD3D_ReceiverZ = 25;      // Receiver Z position

// Derived constants
constexpr int kFDTD3D_GridNX = kFDTD3D_RoomX + 2;  // +2 for boundary conditions
constexpr int kFDTD3D_GridNY = kFDTD3D_RoomY + 2;
constexpr int kFDTD3D_GridNZ = kFDTD3D_RoomZ + 2;

constexpr float kFDTD3D_TimeStep = kFDTD3D_CFLNumber * kFDTD3D_SpatialStep /
                                   (kFDTD3D_SoundSpeed * 1.732050808f);  // √3

//==============================================================================
// FDTD3D Data Structures
//==============================================================================

// FDTD3D grid structure for GPU memory management
struct FDTD3DGrid {
    float* d_pressure;      // Pressure field [nx×ny×nz]
    float* d_velocity_x;    // X-velocity field [(nx+1)×ny×nz]
    float* d_velocity_y;    // Y-velocity field [nx×(ny+1)×nz]
    float* d_velocity_z;    // Z-velocity field [nx×ny×(nz+1)]

    float* h_pressure;      // Host pressure buffer (for verification)
    float* h_velocity_x;    // Host velocity buffers
    float* h_velocity_y;
    float* h_velocity_z;

    size_t pressure_size;   // Memory sizes in bytes
    size_t velocity_x_size;
    size_t velocity_y_size;
    size_t velocity_z_size;

    int nx, ny, nz;         // Grid dimensions
};

// FDTD3D simulation parameters structure
struct FDTD3DParams {
    int nx, ny, nz;                    // Grid dimensions
    float sound_speed;                 // Speed of sound (m/s)
    float spatial_step;                // Grid spacing (m)
    float time_step;                   // Time step (s)
    float air_density;                 // Air density (kg/m³)
    float absorption_coeff;            // Wall absorption coefficient

    int source_x, source_y, source_z;        // Source position
    int receiver_x, receiver_y, receiver_z;  // Receiver position

    int buffer_size;                   // Audio buffer size
    int track_count;                   // Number of tracks
    int steps_per_sample;              // FDTD steps per audio sample

    // Derived coefficients for efficient computation
    float dt_over_rho_dx;              // Δt/(ρ₀×Δx) for velocity update
    float rho_c2_dt_over_dx;           // ρ₀c²Δt/Δx for pressure update
};

class FDTD3DBenchmark : public GPUABenchmark {
public:
    // ============================================================================
    // Constructor/Destructor
    // ============================================================================

    FDTD3DBenchmark(size_t buffer_size = BUFSIZE, size_t track_count = NTRACKS);
    ~FDTD3DBenchmark();

    // ============================================================================
    // Required Implementation from GPUABenchmark
    // ============================================================================

    void setupBenchmark() override;
    void runKernel() override;
    void performBenchmarkIteration() override;
    void validate(ValidationData& validation_data) override;

private:
    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    void allocateFDTD3DGrids();
    void initializeFDTD3DParams();
    void initializeFDTD3DGrid();
    void runFDTD3DTimeStep(int sample_index);
    void calculateCPUReference();
    void cleanupFDTD3DGrids();

    // CPU reference FDTD implementation
    void fdtd3DCPUReference(const float* input_signal, float* output_buffer,
                           const FDTD3DParams& params);

    // Memory size calculation helpers
    size_t fdtd3d_pressure_size(int nx, int ny, int nz);
    size_t fdtd3d_velocity_x_size(int nx, int ny, int nz);
    size_t fdtd3d_velocity_y_size(int nx, int ny, int nz);
    size_t fdtd3d_velocity_z_size(int nx, int ny, int nz);

    // ============================================================================
    // Private Member Variables
    // ============================================================================

    // FDTD3D specific structures
    FDTD3DGrid* fdtd3d_grid = nullptr;
    FDTD3DParams* h_fdtd3d_params = nullptr;
    FDTD3DParams* d_fdtd3d_params = nullptr;

    // Input/output buffers (not using base class standard buffers)
    float* h_input_signal = nullptr;
    float* d_input_signal = nullptr;
    float* h_output_buffer = nullptr;
    float* d_output_buffer = nullptr;

    // CPU reference buffers
    float* cpu_reference = nullptr;
    FDTD3DGrid* cpu_grid = nullptr;

    // Buffer size calculations
    size_t input_signal_bytes;
    size_t output_buffer_bytes;
};

//==============================================================================
// CUDA Kernel Declarations
//==============================================================================

__global__ void fdtd3d_update_velocity_kernel(float* pressure,
                                              float* velocity_x,
                                              float* velocity_y,
                                              float* velocity_z,
                                              int nx, int ny, int nz,
                                              float dt_over_rho_dx);

__global__ void fdtd3d_update_pressure_kernel(float* pressure,
                                              const float* velocity_x,
                                              const float* velocity_y,
                                              const float* velocity_z,
                                              int nx, int ny, int nz,
                                              float rho_c2_dt_over_dx,
                                              float absorption_coeff);

__global__ void fdtd3d_inject_source_kernel(float* pressure,
                                            const float* input_audio,
                                            int source_x, int source_y, int source_z,
                                            int nx, int ny, int buffer_size,
                                            int sample_index, int track_count);

__global__ void fdtd3d_extract_output_kernel(const float* pressure,
                                             float* output_audio,
                                             int receiver_x, int receiver_y, int receiver_z,
                                             int nx, int ny, int buffer_size,
                                             int sample_index, int track_count);

//==============================================================================
// Utility Inline Functions
//==============================================================================

// 3D indexing functions for staggered grid layouts (type-safe, debugger-friendly)
// Pressure grid: [nx × ny × nz]
__device__ __forceinline__ int fdtd3d_pressure_idx(int x, int y, int z, int nx, int ny) {
    return z * nx * ny + y * nx + x;
}

// Velocity-X grid: [(nx+1) × ny × nz] (staggered in X direction)
__device__ __forceinline__ int fdtd3d_velocity_x_idx(int x, int y, int z, int nx_plus_1, int ny) {
    return z * nx_plus_1 * ny + y * nx_plus_1 + x;
}

// Velocity-Y grid: [nx × (ny+1) × nz] (staggered in Y direction)
__device__ __forceinline__ int fdtd3d_velocity_y_idx(int x, int y, int z, int nx, int ny_plus_1) {
    return z * nx * ny_plus_1 + y * nx + x;
}

// Velocity-Z grid: [nx × ny × (nz+1)] (staggered in Z direction)
__device__ __forceinline__ int fdtd3d_velocity_z_idx(int x, int y, int z, int nx, int ny) {
    return z * nx * ny + y * nx + x;
}

// Boundary condition check
__device__ __forceinline__ bool is_fdtd3d_boundary(int x, int y, int z, int nx, int ny, int nz) {
    return (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1);
}

