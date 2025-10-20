// GainStats Benchmark - Gain processing with per-track statistics

import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

export class GainStatsBenchmark extends GPUABenchmark {
    constructor(device, bufferSize = 512, trackCount = 128, options = {}) {
        super(device, 'GainStats', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        const candidateGain = Number(options?.gainValue);
        this.gainValue = Number.isFinite(candidateGain) ? candidateGain : 2.0;

        this.inputData = null;
        this.referenceOutput = null;
        this.referenceStats = null;
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/gain_stats.wgsl');
    }

    async setupBuffers() {
        const totalSamples = this.bufferSize * this.trackCount;

        // Input buffer populated with random audio test data
        this.inputData = this.bufferManager.generateAudioTestData(totalSamples, 'random');
        this.inputBuffer = this.createBuffer(
            'input',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input', this.inputData);

        // Output buffer (track-major layout to mirror native implementations)
        this.outputBuffer = this.createBuffer(
            'output',
            totalSamples * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('output', new Float32Array(totalSamples));

        // Statistics buffer stores [mean, max] per track
        this.statsBuffer = this.createBuffer(
            'stats',
            this.trackCount * 2 * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('stats', new Float32Array(this.trackCount * 2));

        // Uniform parameters
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

        this.generateCPUReference();
    }

    generateCPUReference() {
        const totalSamples = this.bufferSize * this.trackCount;
        this.referenceOutput = new Float32Array(totalSamples);
        this.referenceStats = new Float32Array(this.trackCount * 2);

        for (let track = 0; track < this.trackCount; track++) {
            const trackOffset = track * this.bufferSize;
            let sum = 0.0;
            let max = -Number.MAX_VALUE;

            for (let i = 0; i < this.bufferSize; i++) {
                const idx = trackOffset + i;
                const sample = this.inputData[idx];

                this.referenceOutput[idx] = sample * this.gainValue;
                sum += sample;
                if (sample > max) {
                    max = sample;
                }
            }

            const mean = sum / this.bufferSize;
            this.referenceStats[track * 2] = mean;
            this.referenceStats[track * 2 + 1] = max;
        }
    }

    async createBindGroups() {
        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.inputBuffer } },
                { binding: 1, resource: { buffer: this.outputBuffer } },
                { binding: 2, resource: { buffer: this.statsBuffer } },
                { binding: 3, resource: { buffer: this.paramsBuffer } }
            ],
            label: 'GainStats Bind Group'
        });
    }

    async performIteration() {
        const zeroStats = new Float32Array(this.trackCount * 2);
        this.writeBuffer('stats', zeroStats);
        // Workgroup size is 64, dispatch enough workgroups to cover all tracks
        const workgroups = Math.ceil(this.trackCount / 64);
        await this.executeComputePass(workgroups, 1, 1);
    }

    async validate() {
        const audioData = await this.readBuffer('output');
        const statsData = await this.readBuffer('stats');

        const audioMetrics = this.validateOutput(audioData, this.referenceOutput, VALIDATION_TOLERANCE.GAIN);
        const statsMetrics = this.validateOutput(statsData, this.referenceStats, VALIDATION_TOLERANCE.GAIN);

        const passed = audioMetrics.passed && statsMetrics.passed;
        const maxError = Math.max(audioMetrics.maxError, statsMetrics.maxError);
        const meanError = (audioMetrics.meanError + statsMetrics.meanError) / 2;

        return {
            passed,
            maxError,
            meanError,
            tolerance: Math.max(audioMetrics.tolerance, statsMetrics.tolerance),
            message: `${audioMetrics.message} | ${statsMetrics.message}`,
            components: {
                audio: audioMetrics,
                stats: statsMetrics
            }
        };
    }

    getMetadata() {
        const totalSamples = this.bufferSize * this.trackCount;
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'audio_processing',
            description: 'Gain with per-track statistics (mean, max)',
            gain_value: this.gainValue,
            stats_entries_per_track: 2,
            statistics_tracked: ['mean', 'max'],
            workgroup_size: 64,
            workgroups_dispatched: Math.ceil(this.trackCount / 64),
            total_threads: Math.ceil(this.trackCount / 64) * 64,
            active_threads: this.trackCount,
            total_samples: totalSamples,
            total_tracks: this.trackCount,
            operations_per_sample: 4 // multiply, accumulation, comparison, plus bookkeeping
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.inputData = null;
        this.referenceOutput = null;
        this.referenceStats = null;
    }
}
