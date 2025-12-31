import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class Convolution1DBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
        super(device, 'Conv1D', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        const irCandidate = Number(options?.irLength);
        this.irLength = Number.isFinite(irCandidate) ? Math.max(1, Math.floor(irCandidate)) : 512;

        this.useConstantMemory = options?.useConstantMemory !== undefined
            ? Boolean(options.useConstantMemory)
            : true;

        this.windowFunction = typeof options?.windowFunction === 'string'
            ? options.windowFunction.toLowerCase()
            : 'hann';

        this.inputData = null;
        this.impulseData = null;
        this.referenceOutput = null;
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/conv1d.wgsl');
    }

    async setupBuffers() {
        const totalSamples = this.bufferSize * this.trackCount;

        this.inputData = this.bufferManager.generateAudioTestData(totalSamples, 'random');
        this.inputBuffer = this.createBuffer(
            'input',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input', this.inputData);

        this.impulseData = this.generateImpulseResponses();
        this.irBuffer = this.createBuffer(
            'impulse',
            this.trackCount * this.irLength * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('impulse', this.impulseData);

        this.outputBuffer = this.createBuffer(
            'output',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('output', new Float32Array(totalSamples));

        const paramsData = new ArrayBuffer(16);
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.irLength, true);
        paramsView.setUint32(8, this.trackCount, true);
        paramsView.setUint32(12, this.useConstantMemory ? 1 : 0, true);

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        this.generateCPUReference();
    }

    generateImpulseResponses() {
        const data = new Float32Array(this.trackCount * this.irLength);
        const beta = 6.76; // Typical beta for Kaiser window
        const denom = this.besselI0(beta);

        for (let track = 0; track < this.trackCount; track++) {
            const trackOffset = track * this.irLength;
            const frequency = 0.1 + 0.05 * (track / Math.max(1, this.trackCount - 1));
            const center = (this.irLength - 1) / 2;

            for (let tap = 0; tap < this.irLength; tap++) {
                const t = tap - center;
                const window = this.getWindowCoefficient(tap, beta, denom);
                const sinc = t === 0
                    ? 1.0
                    : Math.sin(2 * Math.PI * frequency * t) / (2 * Math.PI * frequency * t);

                data[trackOffset + tap] = window * sinc / this.irLength;
            }
        }

        return data;
    }

    getWindowCoefficient(index, beta, denom) {
        if (this.irLength <= 1) {
            return 1.0;
        }

        const ratio = index / (this.irLength - 1);

        switch (this.windowFunction) {
            case 'hamming':
                return 0.54 - 0.46 * Math.cos(2 * Math.PI * ratio);
            case 'blackman':
                return 0.42 - 0.5 * Math.cos(2 * Math.PI * ratio) + 0.08 * Math.cos(4 * Math.PI * ratio);
            case 'kaiser': {
                const t = 2 * ratio - 1;
                const argument = beta * Math.sqrt(Math.max(0, 1 - t * t));
                return this.besselI0(argument) / denom;
            }
            case 'hann':
            default:
                return 0.5 * (1 - Math.cos(2 * Math.PI * ratio));
        }
    }
    besselI0(x) {
        let result = 1.0;
        let term = 1.0;
        let k = 1;
        const x2Over4 = (x * x) / 4;

        while (k < 25) {
            term *= x2Over4 / (k * k);
            result += term;
            if (term < 1e-8) {
                break;
            }
            k++;
        }

        return result;
    }

    generateCPUReference() {
        const totalSamples = this.bufferSize * this.trackCount;
        this.referenceOutput = new Float32Array(totalSamples);

        for (let track = 0; track < this.trackCount; track++) {
            const inputOffset = track * this.bufferSize;
            const irOffset = track * this.irLength;

            for (let sample = 0; sample < this.bufferSize; sample++) {
                let acc = 0.0;
                const maxTap = Math.min(this.irLength - 1, sample);

                for (let tap = 0; tap <= maxTap; tap++) {
                    const inputIndex = inputOffset + sample - tap;
                    const irIndex = irOffset + tap;
                    acc += this.inputData[inputIndex] * this.impulseData[irIndex];
                }

                const outputIndex = sample * this.trackCount + track;
                this.referenceOutput[outputIndex] = acc;
            }
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.inputBuffer } },
                { binding: 1, resource: { buffer: this.irBuffer } },
                { binding: 2, resource: { buffer: this.outputBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ],
            label: 'Conv1D Bind Group'
        });
    }

    async performIteration() {
        const workgroups = Math.ceil(this.trackCount / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        const gpuOutput = await this.readBuffer('output');
        return this.validateOutput(gpuOutput, this.referenceOutput, VALIDATION_TOLERANCE.CONVOLUTION);
    }

    getMetadata() {
        const totalSamples = this.bufferSize * this.trackCount;
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'dsp_convolution',
            description: '1D convolution with windowed FIR kernels',
            impulse_response_length: this.irLength,
            window_function: this.windowFunction,
            use_constant_memory: this.useConstantMemory,
            output_layout: 'sample-major',
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.trackCount / 64),
            total_threads: Math.ceil(this.trackCount / 64) * 64,
            active_threads: this.trackCount,
            total_samples: totalSamples,
            total_tracks: this.trackCount,
            operations_per_sample: this.irLength * 2
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.inputData = null;
        this.impulseData = null;
        this.referenceOutput = null;
    }
}
