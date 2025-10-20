// DWG1DAccel Benchmark - Optimized digital waveguide synthesis

import { DWG1DNaiveBenchmark } from './DWG1DNaiveBenchmark.js';

export class DWG1DAccelBenchmark extends DWG1DNaiveBenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
        super(device, bufferSize, trackCount, options);
        this.name = 'DWG1DAccel';
        this.accelerationMode = options?.accelerationMode !== undefined
            ? Boolean(options.accelerationMode)
            : true;
        this.supportsSharedOutput = bufferSize <= 1024;
        if (!this.supportsSharedOutput) {
            console.warn('[DWG1DAccel] bufferSize exceeds shared memory optimization limit (1024); falling back to naive shader.');
            this.accelerationMode = false;
        }
    }

    async loadShader() {
        if (!this.accelerationMode || !this.supportsSharedOutput) {
            return super.loadShader();
        }
        return this.loadShaderFromFile('js/shaders/dwg1d_accel.wgsl');
    }

    getMetadata() {
        const base = super.getMetadata();
        return {
            ...base,
            benchmark_type: 'dwg_accel',
            acceleration_enabled: this.accelerationMode && this.supportsSharedOutput
        };
    }
}
