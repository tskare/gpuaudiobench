// Convolution1D Accel Benchmark - Sample-parallel FIR using shared tiling

import { Convolution1DBenchmark } from './Convolution1DBenchmark.js';

export class Convolution1DAccelBenchmark extends Convolution1DBenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
        super(device, bufferSize, trackCount, options);
        this.name = 'Conv1DAccel';
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/conv1d_accel.wgsl');
    }

    async performIteration() {
        const WORKGROUP_SIZE = 64;
        const workgroupsX = Math.ceil(this.bufferSize / WORKGROUP_SIZE);
        await this.executeComputePass(workgroupsX, this.trackCount, 1);
    }

    getMetadata() {
        const base = super.getMetadata();
        return {
            ...base,
            benchmark_type: 'dsp_convolution_accel',
            acceleration_mode: 'sample_parallel'
        };
    }
}
