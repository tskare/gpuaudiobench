// NoOp Benchmark - Measures GPU kernel launch overhead

import { GPUABenchmark } from '../core/GPUABenchmark.js';

export class NoOpBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128) {
    super(device, 'NoOp', bufferSize, trackCount);
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/noop.wgsl');
    }

    async setupBuffers() {
        // Create a minimal dummy buffer (the shader needs at least one buffer)
        // We only need a few elements since we're just measuring launch overhead
        const dummyBuffer = this.createBuffer(
            'dummy',
            256, // 64 floats * 4 bytes
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );

        // Initialize with zeros
        const zeros = new Float32Array(64);
        this.writeBuffer('dummy', zeros);

        // Store buffer reference for bind group creation
        this.dummyBuffer = dummyBuffer;
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.dummyBuffer
                    }
                }
            ],
            label: 'NoOp Bind Group'
        });
    }

    async performIteration() {
        // Use the standard compute pass execution helper
        // NoOp doesn't clear output buffer since it doesn't produce meaningful output
        await this.executeComputePass(1, 1, 1, false);
    }

    async validate() {
        return {
            passed: true,
            maxError: 0.0,
            meanError: 0.0,
            message: 'NoOp benchmark - no validation needed'
        };
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'overhead_measurement',
            description: 'Measures GPU kernel launch overhead with minimal computation',
            workgroup_size: 64,
            workgroups_dispatched: 1,
            total_threads: 64
        };
    }
}