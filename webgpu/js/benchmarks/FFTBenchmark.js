// FFT Benchmark - Naive DFT workload mirroring native implementations

import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class FFTBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
    super(device, 'FFT', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        const fftCandidate = Number(options?.fftSize);
        this.fftSize = Number.isFinite(fftCandidate) ? Math.max(8, Math.floor(fftCandidate)) : 1024;
        this.outputSize = Math.floor(this.fftSize / 2) + 1;

        this.inputData = null;
        this.referenceOutput = null;
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/fft.wgsl');
    }

    async setupBuffers() {
        const totalSamples = this.bufferSize * this.trackCount;
        const outputSamples = this.trackCount * this.outputSize * 2;

        // Input buffer (real samples)
        this.inputData = this.bufferManager.generateAudioTestData(totalSamples, 'random');
        this.inputBuffer = this.createBuffer(
            'input',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input', this.inputData);

        // Output buffer (complex interleaved)
        this.outputBuffer = this.createBuffer(
            'output',
            outputSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('output', new Float32Array(outputSamples));

        // Parameters: buffer size, fft size, output size, track count
        const paramsData = new ArrayBuffer(16);
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.fftSize, true);
        paramsView.setUint32(8, this.outputSize, true);
        paramsView.setUint32(12, this.trackCount, true);

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        this.generateCPUReference();
    }

    generateCPUReference() {
        const outputSamples = this.trackCount * this.outputSize * 2;
        this.referenceOutput = new Float32Array(outputSamples);

        const sampleLimit = Math.min(this.bufferSize, this.fftSize);

        for (let track = 0; track < this.trackCount; track++) {
            const inputOffset = track * this.bufferSize;
            const outputOffset = track * this.outputSize * 2;

            for (let bin = 0; bin < this.outputSize; bin++) {
                let sumReal = 0.0;
                let sumImag = 0.0;

                for (let n = 0; n < sampleLimit; n++) {
                    const angle = -2 * Math.PI * bin * n / this.fftSize;
                    const sample = this.inputData[inputOffset + n];
                    sumReal += sample * Math.cos(angle);
                    sumImag += sample * Math.sin(angle);
                }

                const base = outputOffset + bin * 2;
                this.referenceOutput[base] = sumReal;
                this.referenceOutput[base + 1] = sumImag;
            }
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.inputBuffer } },
                { binding: 1, resource: { buffer: this.outputBuffer } },
                { binding: 2, resource: { buffer: this.paramsBuffer } }
            ],
            label: 'FFT Bind Group'
        });
    }

    async performIteration() {
        // Workgroup size is 64, dispatch enough workgroups to cover all tracks
        const workgroups = Math.ceil(this.trackCount / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        const gpuOutput = await this.readBuffer('output');
        return this.validateOutput(gpuOutput, this.referenceOutput, VALIDATION_TOLERANCE.FFT_OPERATIONS);
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'dsp_fft',
            description: 'Naive DFT benchmark (forward transform)',
            fft_size: this.fftSize,
            output_bins: this.outputSize,
            total_ffts: this.trackCount,
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.trackCount / 64),
            total_threads: Math.ceil(this.trackCount / 64) * 64,
            active_threads: this.trackCount,
            operations_per_fft: this.fftSize * this.outputSize * 4 // approximate (2 mul-adds per complex component)
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.inputData = null;
        this.referenceOutput = null;
    }
}
