// Random Memory Read Benchmark - Granular synthesis simulation with non-sequential access patterns

import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class RandomMemoryBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
    super(device, 'RandomMemory', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        // Configuration options
        this.sampleMemorySizeMB = Number.isFinite(Number(options.sampleMemorySize))
            ? Math.max(32, Number(options.sampleMemorySize))
            : 128;
        this.sampleMemorySize = Math.floor((this.sampleMemorySizeMB * 1024 * 1024) / 4); // convert MB -> samples
        this.minLoopLength = Math.max(1, Math.floor(options.minLoopLength ?? 1000));
        this.maxLoopLength = Math.max(this.minLoopLength + 1, Math.floor(options.maxLoopLength ?? 48000));
        this.grainDensity = Number.isFinite(Number(options.grainDensity))
            ? Math.max(0.1, Number(options.grainDensity))
            : 1.0;

        this.validateConfiguration();

        this.playheadPositions = null;
        this.referenceData = null;
        this.sampleData = null;
    }

    validateConfiguration() {
        const minRequiredSamples = this.bufferSize + this.maxLoopLength + 1;
        if (this.sampleMemorySize < minRequiredSamples) {
            const requiredMB = Math.ceil((minRequiredSamples * 4) / (1024 * 1024));
            console.warn(
                `[RandomMemory] sample memory (${this.sampleMemorySizeMB} MB) is too small for buffer/loop configuration. Auto-adjusting to ${requiredMB} MB.`
            );
            this.sampleMemorySizeMB = requiredMB;
            this.sampleMemorySize = Math.floor((this.sampleMemorySizeMB * 1024 * 1024) / 4);
        }

        if (this.maxLoopLength <= this.minLoopLength) {
            console.warn('[RandomMemory] maxLoopLength should exceed minLoopLength; nudging maxLoopLength upwards.');
            this.maxLoopLength = this.minLoopLength + 1;
        }

        if (this.grainDensity < 0.1) {
            console.warn('[RandomMemory] grainDensity too low for meaningful stress test; clamping to 0.1.');
            this.grainDensity = 0.1;
        }
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/random_memory.wgsl');
    }

    async setupBuffers() {
        console.log(`Creating sample memory buffer: ${this.sampleMemorySizeMB} MB`);

        // Create large sample memory buffer with random audio data
        this.sampleData = this.bufferManager.generateAudioTestData(this.sampleMemorySize, 'random');
        this.sampleMemoryBuffer = this.createBuffer(
            'sample_memory',
            this.sampleMemorySize * 4, // 4 bytes per float32
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('sample_memory', this.sampleData);

        // Generate random playhead positions for each track
        this.playheadPositions = new Uint32Array(this.trackCount);
        const maxStartPosition = this.sampleMemorySize - this.maxLoopLength - this.bufferSize;

        for (let i = 0; i < this.trackCount; i++) {
            // Generate random loop parameters
            const loopLength = Math.floor(
                Math.random() * (this.maxLoopLength - this.minLoopLength) + this.minLoopLength
            );
            const maxStart = Math.max(0, maxStartPosition - loopLength);
            const startPosition = Math.floor(Math.random() * (maxStart + 1));

            this.playheadPositions[i] = startPosition;
        }

        // Create playheads buffer
        this.playheadsBuffer = this.createBuffer(
            'playheads',
            this.trackCount * 4, // 4 bytes per uint32
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.playheadsBuffer, 0, this.playheadPositions);

        // Create output buffer
        const totalSamples = this.bufferSize * this.trackCount;
        this.outputBuffer = this.createBuffer(
            'output',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );

        // Clear output buffer
        const zeros = new Float32Array(totalSamples);
        this.writeBuffer('output', zeros);

        // Create parameters buffer
        const paramsData = new ArrayBuffer(16); // 4 uint32s
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, this.bufferSize, true);
        paramsView.setUint32(4, this.trackCount, true);
        paramsView.setUint32(8, this.sampleMemorySize, true);
        paramsView.setUint32(12, 0, true); // padding

        this.paramsBuffer = this.createBuffer(
            'params',
            16,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        // Generate CPU reference for validation
        this.generateCPUReference();
    }

    generateCPUReference() {
        const totalSamples = this.bufferSize * this.trackCount;
        this.referenceData = new Float32Array(totalSamples);

        // Process each track
        for (let track = 0; track < this.trackCount; track++) {
            const outputStart = track * this.bufferSize;
            const playheadStart = this.playheadPositions[track];

            // Process samples for this track
            for (let i = 0; i < this.bufferSize; i++) {
                const samplePosition = playheadStart + i;

                if (samplePosition < this.sampleMemorySize) {
                    this.referenceData[outputStart + i] = this.sampleData[samplePosition];
                } else {
                    this.referenceData[outputStart + i] = 0.0;
                }
            }
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.sampleMemoryBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.playheadsBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.outputBuffer }
                },
                {
                    binding: 3,
                    resource: { buffer: this.paramsBuffer }
                }
            ],
            label: 'Random Memory Bind Group'
        });
    }

    async performIteration() {
        // Use the standard compute pass execution helper
        // Workgroup size is 64, dispatch enough workgroups to cover all tracks
        const workgroups = Math.ceil(this.trackCount / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        return this.createValidationResult('output', this.referenceData, VALIDATION_TOLERANCE.RANDOM_MEMORY);
    }

    getMetadata() {
        const sampleMemoryMB = this.sampleMemorySizeMB;
        const totalMemoryMB = sampleMemoryMB + ((this.bufferSize * this.trackCount * 4) / (1024 * 1024));

        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'memory_bandwidth',
            description: 'Random memory access benchmark - simulates granular synthesis',
            sample_memory_size: this.sampleMemorySize,
            sample_memory_mb: sampleMemoryMB,
            total_memory_mb: totalMemoryMB,
            min_loop_length: this.minLoopLength,
            max_loop_length: this.maxLoopLength,
            grain_density: this.grainDensity,
            access_pattern: 'random_non_coalesced',
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.trackCount / 64),
            total_threads: Math.ceil(this.trackCount / 64) * 64,
            active_threads: this.trackCount,
            total_samples: this.bufferSize * this.trackCount,
            memory_reads_per_iteration: this.bufferSize * this.trackCount,
            estimated_bandwidth_gb_s: this.estimateBandwidth()
        };
    }

    estimateBandwidth() {
        // This will be calculated after benchmark runs
        return 0; // Placeholder, will be updated in metadata
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.referenceData = null;
        this.sampleData = null;
        this.playheadPositions = null;
    }
}
