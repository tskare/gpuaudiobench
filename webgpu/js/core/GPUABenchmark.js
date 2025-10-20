// Base class for WebGPU audio benchmarks

import { Statistics } from './Statistics.js';
import { VALIDATION_TOLERANCE } from './ValidationConstants.js';

export class GPUABenchmark {
    constructor(device, name, bufferSize = 512, trackCount = 128) {
        this.device = device;
        this.name = name;
        this.bufferSize = bufferSize;
        this.trackCount = trackCount;

        // WebGPU resources
        this.shaderModule = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.buffers = new Map();

        // Timing and results
        this.results = null;
        this.isSetup = false;
    }

    // Abstract methods - must be implemented by subclasses
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

    async executeComputePass(workgroupsX = 1, workgroupsY = 1, workgroupsZ = 1, clearOutputFirst = true) {
        if (!this.pipeline || !this.bindGroup) {
            throw new Error(`${this.name} benchmark not properly initialized`);
        }

        if (clearOutputFirst && this.getBuffer('output')) {
            const outputBuffer = this.getBufferInfo('output');
            if (outputBuffer) {
                const zeros = new Uint8Array(outputBuffer.size);
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

        // Submit and wait for completion
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        // Wait for GPU to complete
        await this.device.queue.onSubmittedWorkDone();
    }

    async createValidationResult(bufferName, referenceData, tolerance = VALIDATION_TOLERANCE.DEFAULT) {
        if (!referenceData) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                message: 'No reference data available for validation'
            };
        }

        try {
            // Read GPU buffer data using GPUABenchmark's own readBuffer method
            const data = await this.readBuffer(bufferName);

            // Validate length
            if (data.length !== referenceData.length) {
                return {
                    passed: false,
                    message: `Length mismatch: got ${data.length}, expected ${referenceData.length}`,
                    maxError: Infinity,
                    meanError: Infinity,
                    samplesChecked: referenceData.length,
                    tolerance
                };
            }

            // Calculate error metrics
            let maxError = 0;
            let totalError = 0;
            let errorCount = 0;

            for (let i = 0; i < data.length; i++) {
                const error = Math.abs(data[i] - referenceData[i]);
                maxError = Math.max(maxError, error);
                totalError += error;
                if (error > tolerance) {
                    errorCount++;
                }
            }

            const meanError = totalError / data.length;
            const passed = maxError <= tolerance;

            return {
                passed,
                maxError,
                meanError,
                errorCount,
                message: passed
                    ? `Validation passed (max error: ${maxError.toExponential(3)})`
                    : `Validation failed: ${errorCount} values exceed tolerance ${tolerance}`,
                samplesChecked: referenceData.length,
                tolerance
            };
        } catch (error) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                message: `Validation error: ${error.message}`
            };
        }
    }

    async setup() {
        if (this.isSetup) {
            return;
        }

        try {
            console.log(`Setting up ${this.name} benchmark...`);

            // Load shader
            this.shaderModule = await this.loadShader();

            // Create compute pipeline first
            await this.createPipeline();

            // Setup buffers (which may need the pipeline for bind groups)
            await this.setupBuffers();

            // Create bind groups if not done in setupBuffers
            if (!this.bindGroup) {
                await this.createBindGroups();
            }

            this.isSetup = true;
            console.log(`${this.name} benchmark setup complete`);
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
        // Default implementation - subclasses should override
        // if they need specific bind groups
    }

    async runBenchmark(iterations = 100, warmupIterations = 3) {
        if (!this.isSetup) {
            await this.setup();
        }

        console.log(`Running ${this.name} benchmark: ${iterations} iterations with ${warmupIterations} warmup`);

        const latencies = await this.runIterations(
            iterations,
            warmupIterations,
            async () => {
                await this.performIteration();
            }
        );
        const totalSamples = this.bufferSize * this.trackCount;

        // Calculate statistics
        const stats = this.calculateStatistics(latencies);

        // Run validation if implemented
        const validation = await this.validate();

        // Calculate performance metrics
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
                }
            }
        };

        console.log(`${this.name} benchmark complete:`, {
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
                    console.log(`${this.name}: ${i + 1}/${iterations} iterations complete`);
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

        // Create staging buffer for reading
        const stagingBuffer = this.device.createBuffer({
            size: size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: `${this.name}_${bufferName}_staging`
        });

        // Copy data to staging buffer
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
        this.device.queue.submit([encoder.finish()]);

        // Map and read data
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

        console.log(`Exported ${this.name} results to ${a.download}`);
    }

    cleanup() {
        // Destroy all buffers
        for (const [name, bufferInfo] of this.buffers) {
            bufferInfo.buffer.destroy();
        }
        this.buffers.clear();

        // Reset state
        this.shaderModule = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.isSetup = false;

        console.log(`${this.name} benchmark cleaned up`);
    }

    getResults() {
        return this.results;
    }

    isReady() {
        return this.isSetup && this.device && this.pipeline;
    }
}
