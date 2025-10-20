// DataTransfer Benchmark - Measures data transfer overhead with different input/output ratios

import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class DataTransferBenchmark extends GPUABenchmark {
    constructor(device, inputRatio = 1.0, outputRatio = 1.0, bufferSize = 512, trackCount = 128) {
        const variantName = DataTransferBenchmark.getVariantName(inputRatio, outputRatio);
        super(device, `DataTransfer_${variantName}`, bufferSize, trackCount);

        this.inputRatio = inputRatio;
        this.outputRatio = outputRatio;
        this.bufferManager = new BufferManager(device);

        // Calculate actual buffer sizes
        const baseSize = this.bufferSize * this.trackCount;
        this.inputSize = Math.max(1, Math.floor(baseSize * inputRatio));
        this.outputSize = Math.max(1, Math.floor(baseSize * outputRatio));

        this.referenceData = null;
    }

    static getVariantName(inputRatio, outputRatio) {
        const inputPercent = Math.round(inputRatio * 100);
        const outputPercent = Math.round(outputRatio * 100);
        return `${inputPercent.toString().padStart(2, '0')}${outputPercent.toString().padStart(2, '0')}`;
    }

    static createVariants(device, bufferSize = 512, trackCount = 128) {
        return {
            'datacopy0199': new DataTransferBenchmark(device, 0.01, 0.99, bufferSize, trackCount),
            'datacopy2080': new DataTransferBenchmark(device, 0.20, 0.80, bufferSize, trackCount),
            'datacopy5050': new DataTransferBenchmark(device, 0.50, 0.50, bufferSize, trackCount),
            'datacopy8020': new DataTransferBenchmark(device, 0.80, 0.20, bufferSize, trackCount),
            'datacopy9901': new DataTransferBenchmark(device, 0.99, 0.01, bufferSize, trackCount)
        };
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/datatransfer.wgsl');
    }

    async setupBuffers() {
        // Create input buffer with random data
        const inputData = this.bufferManager.generateAudioTestData(this.inputSize, 'random');
        this.inputBuffer = this.createBuffer(
            'input',
            this.inputSize * 4, // 4 bytes per float32
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input', inputData);

        // Create output buffer
        this.outputBuffer = this.createBuffer(
            'output',
            this.outputSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );

        // Clear output buffer
        const zeros = new Float32Array(this.outputSize);
        this.writeBuffer('output', zeros);

        // Create uniform buffer for parameters
        const paramsData = new ArrayBuffer(16); // 16 bytes for alignment
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.inputSize, true);     // input_size
        paramsView.setUint32(4, this.outputSize, true);    // output_size
        paramsView.setFloat32(8, this.inputRatio, true);   // input_ratio
        paramsView.setFloat32(12, this.outputRatio, true); // output_ratio

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        // Generate CPU reference for validation
        this.generateCPUReference(inputData);
    }

    // Matches the GPU's per-thread accumulation logic
    generateCPUReference(inputData) {
        this.referenceData = new Float32Array(this.outputSize);

        // Match GPU shader logic: divide work among threads
        const totalThreads = 64; // Must match @workgroup_size in shader
        const inputElementsPerThread = Math.ceil(this.inputSize / totalThreads);
        const outputElementsPerThread = Math.ceil(this.outputSize / totalThreads);

        for (let threadId = 0; threadId < totalThreads; threadId++) {
            // Calculate this thread's input range
            const inputStart = threadId * inputElementsPerThread;
            const inputEnd = Math.min(inputStart + inputElementsPerThread, this.inputSize);

            // Calculate this thread's output range
            const outputStart = threadId * outputElementsPerThread;
            const outputEnd = Math.min(outputStart + outputElementsPerThread, this.outputSize);

            // Accumulate this thread's input partition
            let sum = 0.0;
            for (let i = inputStart; i < inputEnd; i++) {
                sum += inputData[i];
            }

            // Write this thread's output value
            const outputValue = sum * 0.000001;
            for (let i = outputStart; i < outputEnd; i++) {
                this.referenceData[i] = outputValue;
            }
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
            label: 'DataTransfer Bind Group'
        });
    }

    async performIteration() {
        // Use the standard compute pass execution helper
        await this.executeComputePass(1, 1, 1);
    }

    async validate() {
        return this.createValidationResult('output', this.referenceData, VALIDATION_TOLERANCE.DATA_TRANSFER);
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'data_transfer',
            description: `Data transfer benchmark - ${this.inputRatio}/${this.outputRatio} input/output ratio`,
            input_ratio: this.inputRatio,
            output_ratio: this.outputRatio,
            input_size: this.inputSize,
            output_size: this.outputSize,
            input_mb: (this.inputSize * 4) / (1024 * 1024),
            output_mb: (this.outputSize * 4) / (1024 * 1024),
            total_mb: ((this.inputSize + this.outputSize) * 4) / (1024 * 1024),
            workgroup_size: 64,
            workgroups_dispatched: 1
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.referenceData = null;
    }
}