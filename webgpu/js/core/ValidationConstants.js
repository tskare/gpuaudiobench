export const VALIDATION_TOLERANCE = {
    DEFAULT: 1e-6,
    GAIN: 1e-5,
    IIR_FILTER: 1e-4,
    DATA_TRANSFER: 1e-4,
    FFT_OPERATIONS: 1e-3,
    FDTD_SIMULATION: 5e-2,
    RANDOM_MEMORY: 1e-6,
    CONVOLUTION: 1e-4,
    MODAL_SYNTHESIS: 1e-4,
    DWG_SYNTHESIS: 5e-3,
};
export function getToleranceRationale(tolerance) {
    const entry = Object.entries(VALIDATION_TOLERANCE).find(([_, val]) => val === tolerance);
    if (!entry) return 'Custom tolerance value';

    const rationales = {
        DEFAULT: 'High precision expected for general operations',
        GAIN: 'Simple arithmetic with minimal rounding error',
        IIR_FILTER: 'Recursive operations accumulate rounding errors',
        DATA_TRANSFER: 'Memory operations with minimal computation',
        FFT_OPERATIONS: 'FFT algorithms introduce rounding differences',
        FDTD_SIMULATION: 'Multiple timesteps and numerical dispersion',
        RANDOM_MEMORY: 'Minimal computation, high precision expected',
        CONVOLUTION: 'Accumulation in convolution sum',
        MODAL_SYNTHESIS: 'Recursive filter banks accumulate errors',
        DWG_SYNTHESIS: 'Multiple reflection and filter stages'
    };

    return rationales[entry[0]] || 'No rationale documented';
}
