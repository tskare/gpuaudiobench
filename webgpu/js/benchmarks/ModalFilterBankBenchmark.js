// Modal Filter Bank Benchmark - Bank of resonant modes accumulated per track

import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

const MODE_PARAM_COUNT = 8;

export class ModalFilterBankBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
        super(device, 'ModalFilterBank', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        const modeCandidate = Number(options?.numModes);
        const defaultModes = 1024 * 1024; // Standard: 1048576 modes
        this.numModes = Number.isFinite(modeCandidate) ? Math.max(1, Math.floor(modeCandidate)) : defaultModes;

        const trackCandidate = Number(options?.outputTracks);
        const defaultTracks = 1; // Standard: 1 output track
        this.outputTracks = Number.isFinite(trackCandidate)
            ? Math.max(1, Math.floor(trackCandidate))
            : defaultTracks;
        this.outputTracks = Math.min(this.outputTracks, this.trackCount);

        const spreadCandidate = Number(options?.frequencySpread);
        this.frequencySpread = Number.isFinite(spreadCandidate) ? Math.max(0.01, spreadCandidate) : 1.0;

        const dampingCandidate = Number(options?.dampingCoeff);
        this.dampingCoeff = Number.isFinite(dampingCandidate) ? Math.max(0.0, dampingCandidate) : 0.01;

        this.modeData = null;
        this.referenceOutput = null;
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/modal_filter_bank.wgsl');
    }

    async setupBuffers() {
        // Mode parameters (amp, freq, phase, stateRe, stateIm, ...)
        this.modeData = this.generateModeParameters();
        this.modeBuffer = this.createBuffer(
            'modes',
            this.modeData.length * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('modes', this.modeData);

        // Output buffer: sample-major layout per track
        const outputSamples = this.outputTracks * this.bufferSize;
        this.outputBuffer = this.createBuffer(
            'output',
            outputSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('output', new Float32Array(outputSamples));

        // Uniform parameters
        const paramsData = new ArrayBuffer(16);
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.numModes, true);
        paramsView.setUint32(8, this.outputTracks, true);
        paramsView.setUint32(12, 0, true);

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        this.generateCPUReference();
    }

    generateModeParameters() {
        const data = new Float32Array(this.numModes * MODE_PARAM_COUNT);
        const rng = Math.random;

        for (let mode = 0; mode < this.numModes; mode++) {
            const base = mode * MODE_PARAM_COUNT;

            const amplitude = rng();
            const baseFreq = (mode + 1) / (this.numModes + 1); // normalised 0..1 range
            const freq = Math.min(0.49, baseFreq * this.frequencySpread);
            const phase = rng() * 2 * Math.PI;

            const stateRe = Math.cos(phase);
            const stateIm = Math.sin(phase);

            data[base + 0] = amplitude;
            data[base + 1] = freq;
            data[base + 2] = phase;
            data[base + 3] = stateRe;
            data[base + 4] = stateIm;
            data[base + 5] = this.dampingCoeff;
            data[base + 6] = 0.0;
            data[base + 7] = 0.0;
        }

        return data;
    }

    generateCPUReference() {
        this.referenceOutput = new Float32Array(this.outputTracks * this.bufferSize);

        for (let mode = 0; mode < this.numModes; mode++) {
            const base = mode * MODE_PARAM_COUNT;
            const amp = this.modeData[base + 0];
            const freq = this.modeData[base + 1];
            let stateRe = this.modeData[base + 3];
            let stateIm = this.modeData[base + 4];

            const outputTrack = mode % this.outputTracks;
            const cosVal = Math.cos(2 * Math.PI * freq);
            const sinVal = Math.sin(2 * Math.PI * freq);

            for (let sample = 0; sample < this.bufferSize; sample++) {
                const newRe = stateRe * cosVal - stateIm * sinVal;
                const newIm = stateRe * sinVal + stateIm * cosVal;
                stateRe = newRe;
                stateIm = newIm;

                const outIndex = outputTrack * this.bufferSize + sample;
                this.referenceOutput[outIndex] += amp * stateRe;
            }
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.modeBuffer } },
                { binding: 1, resource: { buffer: this.outputBuffer } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ],
            label: 'Modal Filter Bank Bind Group'
        });
    }

    async performIteration() {
        const workgroups = Math.ceil(this.numModes / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        const gpuOutput = await this.readBuffer('output');
        const metrics = this.calculateErrorMetrics(gpuOutput, this.referenceOutput, VALIDATION_TOLERANCE.MODAL_SYNTHESIS);
        return {
            passed: metrics.passed,
            maxError: metrics.maxError,
            meanError: metrics.meanError,
            tolerance: metrics.tolerance,
            message: metrics.message
        };
    }

    calculateErrorMetrics(gpuData, referenceData, tolerance) {
        if (!referenceData) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                tolerance,
                message: 'No reference data available'
            };
        }

        if (gpuData.length !== referenceData.length) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                tolerance,
                message: `Length mismatch: got ${gpuData.length}, expected ${referenceData.length}`
            };
        }

        let maxError = 0;
        let totalError = 0;

        for (let i = 0; i < referenceData.length; i++) {
            const error = Math.abs(gpuData[i] - referenceData[i]);
            if (error > maxError) {
                maxError = error;
            }
            totalError += error;
        }

        const meanError = referenceData.length > 0 ? totalError / referenceData.length : 0;
        const passed = maxError <= tolerance;

        return {
            passed,
            maxError,
            meanError,
            tolerance,
            message: passed
                ? `Validation passed (max error ${maxError.toExponential(3)})`
                : `Max error ${maxError.toExponential(3)} exceeds tolerance ${tolerance}`
        };
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'modal_filter_bank',
            description: 'Modal synthesis filter bank with per-mode accumulations',
            num_modes: this.numModes,
            output_tracks: this.outputTracks,
            mode_parameter_count: MODE_PARAM_COUNT,
            frequency_spread: this.frequencySpread,
            damping_coefficient: this.dampingCoeff,
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.numModes / 64),
            total_samples: this.bufferSize * this.outputTracks
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.modeData = null;
        this.referenceOutput = null;
    }
}
