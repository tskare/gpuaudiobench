import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class GainBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128) {
        super(device, 'Gain', bufferSize, trackCount);
        this.gainValue = 2.0;
        this.bufferManager = new BufferManager(device);
        this.referenceData = null;
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/gain.wgsl');
    }

    async setupBuffers() {
        const totalSamples = this.bufferSize * this.trackCount;

        const inputData = this.bufferManager.generateAudioTestData(totalSamples, 'random');
        this.inputBuffer = this.createBuffer(
            'input',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input', inputData);

        this.outputBuffer = this.createBuffer(
            'output',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        );

        const zeros = new Float32Array(totalSamples);
        this.writeBuffer('output', zeros);

        const paramsData = new ArrayBuffer(16);
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.trackCount, true);
        paramsView.setFloat32(8, this.gainValue, true);
        paramsView.setFloat32(12, 0.0, true);

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        this.referenceData = new Float32Array(totalSamples);
        for (let i = 0; i < totalSamples; i++) {
            this.referenceData[i] = inputData[i] * this.gainValue;
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.inputBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.outputBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.paramsBuffer
                    }
                }
            ],
            label: 'Gain Bind Group'
        });
    }

    async performIteration() {
        const workgroups = Math.ceil(this.trackCount / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        return this.createValidationResult('output', this.referenceData, VALIDATION_TOLERANCE.GAIN);
    }

    getMetadata() {
        const workgroups = Math.ceil(this.trackCount / 64);
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'audio_processing',
            description: 'Simple gain processing - multiplies audio samples by fixed gain',
            gain_value: this.gainValue,
            workgroup_size: 64,
            workgroups_dispatched: workgroups,
            total_threads: workgroups * 64,
            active_threads: this.trackCount,
            total_samples: this.bufferSize * this.trackCount,
            operations_per_sample: 1, // One multiplication per sample
            total_operations: this.bufferSize * this.trackCount
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.referenceData = null;
    }
}
