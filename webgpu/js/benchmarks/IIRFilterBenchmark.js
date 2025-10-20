// IIR Filter Benchmark - 2nd order biquad digital filter (Direct Form II)

import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class IIRFilterBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128) {
        super(device, 'IIRFilter', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        // Filter coefficients for Butterworth lowpass at fs/4, Q=0.707
        this.coefficients = this.calculateButterworthCoefficients();
        this.referenceData = null;
        this.inputData = null;
    }

    // Calculate Butterworth lowpass filter coefficients (fs/4, Q=0.707)
    calculateButterworthCoefficients() {
        const omega = Math.PI / 2;  // fs/4 normalized frequency
        const sin_omega = Math.sin(omega);
        const cos_omega = Math.cos(omega);
        const alpha = sin_omega / Math.sqrt(2);  // Q = 0.707

        const a0 = 1 + alpha;
        const b0 = ((1 - cos_omega) / 2) / a0;
        const b1 = (1 - cos_omega) / a0;
        const b2 = ((1 - cos_omega) / 2) / a0;
        const a1 = (-2 * cos_omega) / a0;
        const a2 = (1 - alpha) / a0;

        return { b0, b1, b2, a1, a2 };
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/iir_filter.wgsl');
    }

    async setupBuffers() {
        const totalSamples = this.bufferSize * this.trackCount;

        // Create input buffer with random audio data
        this.inputData = this.bufferManager.generateAudioTestData(totalSamples, 'random');
        this.inputBuffer = this.createBuffer(
            'input',
            totalSamples * 4, // 4 bytes per float32
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input', this.inputData);

        // Create output buffer
        this.outputBuffer = this.createBuffer(
            'output',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );

        // Clear output buffer
        const zeros = new Float32Array(totalSamples);
        this.writeBuffer('output', zeros);

        // Create coefficients buffer
        const coeffData = new ArrayBuffer(32); // 8 floats * 4 bytes (with padding)
        const coeffView = new Float32Array(coeffData);
        coeffView[0] = this.coefficients.b0;
        coeffView[1] = this.coefficients.b1;
        coeffView[2] = this.coefficients.b2;
        coeffView[3] = this.coefficients.a1;
        coeffView[4] = this.coefficients.a2;
        // coeffView[5-7] remain 0 (padding)

        this.coefficientsBuffer = this.createBuffer(
            'coefficients',
            32,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.coefficientsBuffer, 0, coeffData);

        // Create filter states buffer (2 states per track)
        const statesSize = this.trackCount * 2;
        const statesData = new Float32Array(statesSize); // Initialize to zeros
        this.statesBuffer = this.createBuffer(
            'states',
            statesSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        );
        this.writeBuffer('states', statesData);

        // Create parameters buffer
        const paramsData = new ArrayBuffer(16); // 4 uints with padding
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.trackCount, true);
        // bytes 8-15 remain 0 (padding)

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        // Generate CPU reference for validation
        this.generateCPUReference();
    }

    generateCPUReference() {
        const totalSamples = this.bufferSize * this.trackCount;
        this.referenceData = new Float32Array(totalSamples);

        // CPU filter states (2 per track)
        const cpuStates = new Float32Array(this.trackCount * 2);

        // Process each track
        for (let track = 0; track < this.trackCount; track++) {
            const startIdx = track * this.bufferSize;
            const stateIdx = track * 2;
            let z1 = cpuStates[stateIdx];
            let z2 = cpuStates[stateIdx + 1];

            // Process samples in this track
            for (let i = 0; i < this.bufferSize; i++) {
                const sampleIdx = startIdx + i;
                const x = this.inputData[sampleIdx];

                // Direct Form II biquad
                const w = x - this.coefficients.a1 * z1 - this.coefficients.a2 * z2;
                const y = this.coefficients.b0 * w + this.coefficients.b1 * z1 + this.coefficients.b2 * z2;

                // Update states
                z2 = z1;
                z1 = w;

                // Store output
                this.referenceData[sampleIdx] = y;
            }

            // Save states for this track
            cpuStates[stateIdx] = z1;
            cpuStates[stateIdx + 1] = z2;
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.inputBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.outputBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.coefficientsBuffer }
                },
                {
                    binding: 3,
                    resource: { buffer: this.statesBuffer }
                },
                {
                    binding: 4,
                    resource: { buffer: this.paramsBuffer }
                }
            ],
            label: 'IIR Filter Bind Group'
        });
    }

    async performIteration() {
        // Reset filter states for clean iteration (in addition to output buffer clearing)
        const statesZeros = new Float32Array(this.trackCount * 2);
        this.writeBuffer('states', statesZeros);

        // Use the standard compute pass execution helper
        // Workgroup size is 64, dispatch enough workgroups to cover all tracks
        const workgroups = Math.ceil(this.trackCount / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        return this.createValidationResult('output', this.referenceData, VALIDATION_TOLERANCE.IIR_FILTER);
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'dsp_filter',
            description: 'IIR biquad filter - 2nd order Butterworth lowpass at fs/4',
            filter_type: 'biquad_lowpass',
            cutoff_frequency: 'fs/4',
            q_factor: 0.707,
            coefficients: this.coefficients,
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.trackCount / 64),
            total_threads: Math.ceil(this.trackCount / 64) * 64,
            active_threads: this.trackCount,
            total_samples: this.bufferSize * this.trackCount,
            operations_per_sample: 5, // 5 multiplications + 4 additions per sample
            total_operations: this.bufferSize * this.trackCount * 9
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.referenceData = null;
        this.inputData = null;
    }
}