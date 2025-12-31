import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

const WAVEGUIDE_STATE_BYTES = 32;

export class DWG1DNaiveBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
        super(device, 'DWG1DNaive', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        this.numWaveguides = Math.max(1, trackCount);

        const lengthCandidate = Number(options?.waveguideLength);
        this.waveguideLength = Number.isFinite(lengthCandidate) ? Math.max(4, Math.floor(lengthCandidate)) : 256;

        const inputTapRatio = Number(options?.inputTapPos);
        this.inputTapRatio = Number.isFinite(inputTapRatio) ? this.clamp01(inputTapRatio) : 0.1;

        const outputTapRatio = Number(options?.outputTapPos);
        this.outputTapRatio = Number.isFinite(outputTapRatio) ? this.clamp01(outputTapRatio) : 0.9;

        const reflection = Number(options?.reflectionCoeff);
        this.reflectionCoeff = Number.isFinite(reflection) ? reflection : -0.99;

        const damping = Number(options?.dampingFactor);
        this.dampingCoeff = Number.isFinite(damping) ? damping : 0.9999;

        this.modeStatesCPU = null;
        this.inputSignal = null;
        this.referenceOutput = null;
        this.delayForwardCPU = null;
        this.delayBackwardCPU = null;
    }

    clamp01(value) {
        return Math.min(1, Math.max(0, value));
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/dwg1d_naive.wgsl');
    }

    async setupBuffers() {
        const stateBuffer = this.generateWaveguideStates();
        this.waveguideBuffer = this.createBuffer(
            'waveguides',
            stateBuffer.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.waveguideBuffer, 0, stateBuffer);

        const delaySize = this.numWaveguides * this.waveguideLength;
        this.delayForwardData = new Float32Array(delaySize);
        this.delayBackwardData = new Float32Array(delaySize);

        this.delayForwardBuffer = this.createBuffer(
            'delayForward',
            delaySize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.delayBackwardBuffer = this.createBuffer(
            'delayBackward',
            delaySize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('delayForward', this.delayForwardData);
        this.writeBuffer('delayBackward', this.delayBackwardData);

        this.inputSignal = new Float32Array(this.bufferSize);
        if (this.bufferSize > 0) {
            this.inputSignal[0] = 1.0;
        }
        for (let i = 1; i < this.bufferSize; i++) {
            this.inputSignal[i] = (Math.random() - 0.5) * 0.05;
        }
        this.inputSignalBuffer = this.createBuffer(
            'inputSignal',
            this.bufferSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('inputSignal', this.inputSignal);

        this.outputBuffer = this.createBuffer(
            'output',
            this.bufferSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('output', new Float32Array(this.bufferSize));

        const paramsData = new ArrayBuffer(16);
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.waveguideLength, true);
        paramsView.setUint32(8, this.numWaveguides, true);
        paramsView.setUint32(12, 0, true);

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        this.generateCPUReference();
    }

    generateWaveguideStates() {
        const buffer = new ArrayBuffer(this.numWaveguides * WAVEGUIDE_STATE_BYTES);
        const view = new DataView(buffer);
        this.modeStatesCPU = [];

        for (let i = 0; i < this.numWaveguides; i++) {
            const length = this.waveguideLength;
            const inputTap = Math.floor(this.inputTapRatio * (length - 1));
            const outputTap = Math.floor(this.outputTapRatio * (length - 1));
            const gain = 0.5 + Math.random() * 0.5;

            const offset = i * WAVEGUIDE_STATE_BYTES;
            view.setUint32(offset + 0, length, true);
            view.setUint32(offset + 4, inputTap, true);
            view.setUint32(offset + 8, outputTap, true);
            view.setUint32(offset + 12, 0, true); // writePos
            view.setFloat32(offset + 16, gain, true);
            view.setFloat32(offset + 20, this.reflectionCoeff, true);
            view.setFloat32(offset + 24, this.dampingCoeff, true);
            view.setFloat32(offset + 28, 0.0, true);

            this.modeStatesCPU.push({
                length,
                inputTap,
                outputTap,
                gain,
                reflection: this.reflectionCoeff,
                damping: this.dampingCoeff
            });
        }

        return buffer;
    }

    generateCPUReference() {
        const forward = new Float32Array(this.delayForwardData.length);
        const backward = new Float32Array(this.delayBackwardData.length);
        const output = new Float32Array(this.bufferSize);

        for (let wgIndex = 0; wgIndex < this.numWaveguides; wgIndex++) {
            const state = this.modeStatesCPU[wgIndex];
            const length = state.length;
            if (length === 0) continue;

            const base = wgIndex * this.waveguideLength;
            const halfLength = Math.floor(length / 2);

            for (let sample = 0; sample < this.bufferSize; sample++) {
                const inputValue = this.inputSignal[sample] * state.gain;
                const writeIndex = (sample) % length; // writePos is zero

                const forwardIndex = base + writeIndex;
                const backwardIndex = base + ((sample + halfLength) % length);

                let forwardSample = forward[forwardIndex] * state.damping;
                let backwardSample = backward[backwardIndex] * state.damping;

                if (writeIndex === state.inputTap) {
                    forwardSample += inputValue;
                    backwardSample += inputValue;
                }

                const newForward = backwardSample * state.reflection + inputValue;
                const newBackward = forwardSample * state.reflection + inputValue;

                forward[forwardIndex] = newForward;
                backward[backwardIndex] = newBackward;

                if (writeIndex === state.outputTap) {
                    output[sample] += (forwardSample + backwardSample) * 0.5;
                }
            }
        }

        this.referenceOutput = output;
        this.delayForwardCPU = forward;
        this.delayBackwardCPU = backward;
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.waveguideBuffer } },
                { binding: 1, resource: { buffer: this.delayForwardBuffer } },
                { binding: 2, resource: { buffer: this.delayBackwardBuffer } },
                { binding: 3, resource: { buffer: this.inputSignalBuffer } },
                { binding: 4, resource: { buffer: this.outputBuffer } },
                { binding: 5, resource: { buffer: this.paramsBuffer } }
            ],
            label: 'DWG1D Naive Bind Group'
        });
    }

    async performIteration() {
        const zeroDelays = this.getZeroedArray('f32', this.delayForwardData.length);
        const zeroOutput = this.getZeroedArray('f32', this.bufferSize);
        this.writeBuffer('delayForward', zeroDelays);
        this.writeBuffer('delayBackward', zeroDelays);
        this.writeBuffer('output', zeroOutput);

        const workgroups = Math.ceil(this.numWaveguides / 64);
        await this.executeComputePass(workgroups, 1, 1, false);
    }

    async validate() {
        const gpuOutput = await this.readBuffer('output');
        return this.validateOutput(gpuOutput, this.referenceOutput, VALIDATION_TOLERANCE.DWG_SYNTHESIS);
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'dwg_naive',
            description: 'Naive digital waveguide synthesis',
            num_waveguides: this.numWaveguides,
            waveguide_length: this.waveguideLength,
            input_tap_ratio: this.inputTapRatio,
            output_tap_ratio: this.outputTapRatio,
            reflection_coeff: this.reflectionCoeff,
            damping_coeff: this.dampingCoeff,
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.numWaveguides / 64),
            total_samples: this.bufferSize
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.modeStatesCPU = null;
        this.inputSignal = null;
        this.referenceOutput = null;
        this.delayForwardCPU = null;
        this.delayBackwardCPU = null;
    }
}
