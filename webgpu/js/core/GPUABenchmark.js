import { Statistics } from './Statistics.js';
import { VALIDATION_TOLERANCE } from './ValidationConstants.js';

export class GPUABenchmark {
    constructor(device, name, bufferSize = 512, trackCount = 128) {
        this.device = device;
        this.name = name;
        this.bufferSize = bufferSize;
        this.trackCount = trackCount;

        this.shaderModule = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.buffers = new Map();

        this.results = null;
        this.isSetup = false;

        this.quiet = false;
        this.zeroBufferCache = new Map();
        this.totalAllocatedBytes = 0;
        this.peakAllocatedBytes = 0;
    }

    async loadShader() {
        throw new Error(`${this.name}: loadShader() must be implemented`);
    }

    async setupBuffers() {
        throw new Error(`${this.name}: setupBuffers() must be implemented`);
    }

    async performIteration() {
        throw new Error(`${this.name}: performIteration() must be implemented`);
    }

    async validate() {
        return {
            passed: true,
            maxError: 0.0,
            meanError: 0.0,
            message: 'No validation implemented'
        };
    }

    async loadShaderFromFile(shaderPath, label = null) {
        try {
            const response = await fetch(shaderPath);
            if (!response.ok) {
                throw new Error(`Failed to load shader: ${response.status} ${response.statusText}`);
            }
            const shaderCode = await response.text();
            return this.device.createShaderModule({
                code: shaderCode,
                label: label || `${this.name} Shader`
            });
        } catch (error) {
            console.error(`Failed to load ${label || this.name} shader:`, error);
            throw error;
        }
    }

    createBufferWithData(name, data, usage) {
        const buffer = this.createBuffer(name, data.byteLength, usage);
        this.writeBuffer(name, data);
        return buffer;
    }

    createZeroedBuffer(name, sizeInBytes, usage) {
        const buffer = this.createBuffer(name, sizeInBytes, usage);
        const zeros = new Float32Array(sizeInBytes / 4);
        this.writeBuffer(name, zeros);
        return buffer;
    }

    createUniformBuffer(name, data, alignedSize = 16) {
        const buffer = this.createBuffer(
            name,
            alignedSize,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    }

    validateOutput(gpuData, referenceData, tolerance) {
        return Statistics.calculateErrorMetrics(gpuData, referenceData, tolerance);
    }

    setQuiet(enabled) {
        this.quiet = Boolean(enabled);
    }

    log(...args) {
        if (!this.quiet) {
            console.log(...args);
        }
    }

    getZeroedArray(kind, size) {
        const key = `${kind}:${size}`;
        if (this.zeroBufferCache.has(key)) {
            return this.zeroBufferCache.get(key);
        }

        let zeros;
        if (kind === 'f32') {
            zeros = new Float32Array(size);
        } else {
            zeros = new Uint8Array(size);
        }

        // Limit cache size to prevent unbounded growth
        if (this.zeroBufferCache.size >= 16) {
            const firstKey = this.zeroBufferCache.keys().next().value;
            this.zeroBufferCache.delete(firstKey);
        }
        this.zeroBufferCache.set(key, zeros);
        return zeros;
    }

    async executeComputePass(workgroupsX = 1, workgroupsY = 1, workgroupsZ = 1, clearOutputFirst = true) {
        if (!this.pipeline || !this.bindGroup) {
            throw new Error(`${this.name} benchmark not properly initialized`);
        }

        if (clearOutputFirst && this.getBuffer('output')) {
            const outputBuffer = this.getBufferInfo('output');
            if (outputBuffer) {
                const zeros = this.getZeroedArray('u8', outputBuffer.size);
                this.device.queue.writeBuffer(outputBuffer.buffer, 0, zeros);
            }
        }

        const encoder = this.device.createCommandEncoder({
            label: `${this.name} Command Encoder`
        });

        const computePass = encoder.beginComputePass({
            label: `${this.name} Compute Pass`
        });

        computePass.setPipeline(this.pipeline);
        computePass.setBindGroup(0, this.bindGroup);

        computePass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);

        computePass.end();

        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        await this.device.queue.onSubmittedWorkDone();
    }
    async createValidationResult(bufferName, referenceData, tolerance = VALIDATION_TOLERANCE.DEFAULT) {
        if (!referenceData) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                errorCount: 0,
                samplesChecked: 0,
                tolerance,
                message: 'No reference data available for validation'
            };
        }

        try {
            const data = await this.readBuffer(bufferName);

            return this.validateOutput(data, referenceData, tolerance);
        } catch (error) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                errorCount: 0,
                samplesChecked: 0,
                tolerance,
                message: `Validation error: ${error.message}`
            };
        }
    }
    async setup() {
        if (this.isSetup) {
            return;
        }

        try {
            this.log(`Setting up ${this.name} benchmark...`);

            this.shaderModule = await this.loadShader();

            await this.createPipeline();

            await this.setupBuffers();

            if (!this.bindGroup) {
                await this.createBindGroups();
            }

            this.isSetup = true;
            this.log(`${this.name} benchmark setup complete`);
        } catch (error) {
            console.error(`Failed to setup ${this.name} benchmark:`, error);
            throw error;
        }
    }
    async createPipeline() {
        this.pipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'main'
            }
        });
    }
    async createBindGroups() {
    }
    async runBenchmark(iterations = 100, warmupIterations = 3) {
        if (!this.isSetup) {
            await this.setup();
        }

        this.log(`Running ${this.name} benchmark: ${iterations} iterations with ${warmupIterations} warmup`);

        const latencies = await this.runIterations(
            iterations,
            warmupIterations,
            async () => {
                await this.performIteration();
            }
        );
        const totalSamples = this.bufferSize * this.trackCount;

        const stats = this.calculateStatistics(latencies);

        const validation = await this.validate();

        const bytesProcessed = totalSamples * 4; // 4 bytes per float32
        const meanLatencyMs = stats.mean;
        const throughputGBps = (bytesProcessed / (1024 * 1024 * 1024)) / (meanLatencyMs / 1000);
        const samplesPerSec = totalSamples / (meanLatencyMs / 1000);

        this.results = {
            benchmark: this.name,
            latencies,
            statistics: stats,
            validation,
            metadata: {
                bufferSize: this.bufferSize,
                trackCount: this.trackCount,
                totalSamples,
                iterations,
                warmupIterations,
                timestamp: Date.now(),
                performance: {
                    throughputGBps: throughputGBps.toFixed(3),
                    samplesPerSec: samplesPerSec.toFixed(0),
                    bytesProcessed,
                    meanLatencyMs: meanLatencyMs.toFixed(3)
                },
                memory: this.getMemorySummary()
            }
        };

        this.log(`${this.name} benchmark complete:`, {
            median: `${stats.median.toFixed(3)}ms`,
            p95: `${stats.p95.toFixed(3)}ms`,
            max: `${stats.max.toFixed(3)}ms`,
            validation: validation.passed ? 'PASSED' : 'FAILED'
        });

        return this.results;
    }
    calculateStatistics(latencies) {
        return Statistics.calculate(latencies);
    }

    async runIterations(iterations, warmupIterations, iterationFn) {
        for (let i = 0; i < warmupIterations; i++) {
            await iterationFn(i, true);
        }

        const latencies = [];
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await iterationFn(i, false);
            const end = performance.now();
            latencies.push(end - start);

            if (iterations > 50) {
                const interval = Math.floor(iterations / 10);
                if (interval > 0 && (i + 1) % interval === 0) {
                    this.log(`${this.name}: ${i + 1}/${iterations} iterations complete`);
                }
            }
        }
        return latencies;
    }
    createBuffer(name, size, usage) {
        const buffer = this.device.createBuffer({
            size,
            usage,
            label: `${this.name}_${name}`
        });
        this.buffers.set(name, {
            buffer,
            size,
            usage
        });
        this.totalAllocatedBytes += size;
        this.peakAllocatedBytes = Math.max(this.peakAllocatedBytes, this.totalAllocatedBytes);
        return buffer;
    }
    getBuffer(name) {
        const bufferInfo = this.buffers.get(name);
        if (!bufferInfo) {
            throw new Error(`Buffer '${name}' not found in ${this.name} benchmark`);
        }
        return bufferInfo.buffer;
    }
    getBufferInfo(name) {
        return this.buffers.get(name);
    }
    writeBuffer(bufferName, data) {
        const buffer = this.getBuffer(bufferName);
        this.device.queue.writeBuffer(buffer, 0, data);
    }
    async readBuffer(bufferName) {
        const bufferInfo = this.getBufferInfo(bufferName);
        if (!bufferInfo) {
            throw new Error(`Buffer '${bufferName}' not found in ${this.name} benchmark`);
        }
        const buffer = bufferInfo.buffer;
        const size = bufferInfo.size;

        const stagingBuffer = this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: `${this.name}_${bufferName}_staging`
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
        this.device.queue.submit([encoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange().slice());
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        return data;
    }
    exportResults(filename) {
        if (!this.results) {
            throw new Error('No benchmark results to export');
        }

        const data = {
            ...this.results,
            exportTimestamp: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || `${this.name}_benchmark_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.log(`Exported ${this.name} results to ${a.download}`);
    }
    cleanup() {
        for (const [name, bufferInfo] of this.buffers) {
            bufferInfo.buffer.destroy();
            this.totalAllocatedBytes = Math.max(0, this.totalAllocatedBytes - bufferInfo.size);
        }
        this.buffers.clear();

        this.shaderModule = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.isSetup = false;
        this.zeroBufferCache.clear();

        this.log(`${this.name} benchmark cleaned up`);
    }
    getResults() {
        return this.results;
    }
    isReady() {
        return this.isSetup && this.device && this.pipeline;
    }

    getMemorySummary() {
        return {
            totalBytes: this.totalAllocatedBytes,
            peakBytes: this.peakAllocatedBytes,
            totalBuffers: this.buffers.size
        };
    }
}
