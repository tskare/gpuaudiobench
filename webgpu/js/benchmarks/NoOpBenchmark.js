import { GPUABenchmark } from '../core/GPUABenchmark.js';

export class NoOpBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128) {
        super(device, 'NoOp', bufferSize, trackCount);
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/noop.wgsl');
    }

    async setupBuffers() {
        const dummyBuffer = this.createBuffer(
            'dummy',
            256, // 64 floats * 4 bytes
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );

        const zeros = new Float32Array(64);
        this.writeBuffer('dummy', zeros);

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
        await this.executeComputePass(1, 1, 1, false);
    }
    async validate() {
        return {
            passed: true,
            maxError: 0.0,
            meanError: 0.0,
            errorCount: 0,
            samplesChecked: 0,
            tolerance: 0,
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
