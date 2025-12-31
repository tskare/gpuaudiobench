export class BufferManager {
    constructor(device) {
        this.device = device;
        this.buffers = new Map();
        this.totalAllocated = 0;
    }
    createBuffer(name, size, usage, label = null) {
        if (this.buffers.has(name)) {
            console.warn(`Buffer '${name}' already exists, replacing it`);
            this.destroyBuffer(name);
        }

        const buffer = this.device.createBuffer({
            size,
            usage,
            label: label || name
        });

        this.buffers.set(name, {
            buffer,
            size,
            usage,
            created: Date.now()
        });

        this.totalAllocated += size;
        console.log(`Created buffer '${name}': ${this.formatBytes(size)}`);

        return buffer;
    }
    createAudioBuffer(name, sampleCount, testPattern = 'random') {
        const size = sampleCount * 4; // 4 bytes per float32
        const buffer = this.createBuffer(
            name,
            size,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );

        const data = this.generateAudioTestData(sampleCount, testPattern);
        this.device.queue.writeBuffer(buffer, 0, data);

        return buffer;
    }
    generateAudioTestData(sampleCount, pattern = 'random') {
        const data = new Float32Array(sampleCount);

        switch (pattern) {
            case 'random':
                for (let i = 0; i < sampleCount; i++) {
                    data[i] = (Math.random() - 0.5) * 2.0; // Random between -1 and 1
                }
                break;

            case 'sine':
                const frequency = 440; // A4 note
                const sampleRate = 48000;
                for (let i = 0; i < sampleCount; i++) {
                    data[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5;
                }
                break;

            case 'impulse':
                data.fill(0);
                if (sampleCount > 0) data[0] = 1.0;
                break;

            case 'step':
                data.fill(0.5);
                break;

            case 'ramp':
                for (let i = 0; i < sampleCount; i++) {
                    data[i] = (i / sampleCount) * 2.0 - 1.0; // Ramp from -1 to 1
                }
                break;

            case 'zeros':
                data.fill(0);
                break;

            case 'ones':
                data.fill(1);
                break;

            default:
                console.warn(`Unknown test pattern '${pattern}', using random`);
                return this.generateAudioTestData(sampleCount, 'random');
        }

        return data;
    }
    createMultiTrackBuffers(baseName, bufferSize, trackCount, pattern = 'random') {
        const totalSamples = bufferSize * trackCount;

        const inputBuffer = this.createAudioBuffer(
            `${baseName}_input`,
            totalSamples,
            pattern
        );

        const outputBuffer = this.createBuffer(
            `${baseName}_output`,
            totalSamples * 4, // 4 bytes per float32
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );

        return { inputBuffer, outputBuffer };
    }
    getBuffer(name) {
        const bufferInfo = this.buffers.get(name);
        if (!bufferInfo) {
            throw new Error(`Buffer '${name}' not found`);
        }
        return bufferInfo.buffer;
    }
    getBufferInfo(name) {
        return this.buffers.get(name);
    }
    writeBuffer(name, data, offset = 0) {
        const buffer = this.getBuffer(name);
        this.device.queue.writeBuffer(buffer, offset, data);
    }
    async readBuffer(name) {
        const bufferInfo = this.getBufferInfo(name);
        if (!bufferInfo) {
            throw new Error(`Buffer '${name}' not found`);
        }

        const stagingBuffer = this.device.createBuffer({
            size: bufferInfo.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            label: `${name}_staging`
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(bufferInfo.buffer, 0, stagingBuffer, 0, bufferInfo.size);
        this.device.queue.submit([encoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange().slice());
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        return data;
    }
    copyBuffer(sourceName, destName, size = null) {
        const sourceBuffer = this.getBuffer(sourceName);
        const destBuffer = this.getBuffer(destName);
        const sourceInfo = this.getBufferInfo(sourceName);

        const copySize = size || sourceInfo.size;

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(sourceBuffer, 0, destBuffer, 0, copySize);
        this.device.queue.submit([encoder.finish()]);
    }
    clearBuffer(name) {
        const bufferInfo = this.getBufferInfo(name);
        if (!bufferInfo) {
            throw new Error(`Buffer '${name}' not found`);
        }

        const zeros = new Uint8Array(bufferInfo.size);
        this.device.queue.writeBuffer(bufferInfo.buffer, 0, zeros);
    }
    destroyBuffer(name) {
        const bufferInfo = this.buffers.get(name);
        if (bufferInfo) {
            bufferInfo.buffer.destroy();
            this.totalAllocated -= bufferInfo.size;
            this.buffers.delete(name);
            console.log(`Destroyed buffer '${name}': ${this.formatBytes(bufferInfo.size)}`);
        }
    }
    destroyAll() {
        for (const [name, bufferInfo] of this.buffers) {
            bufferInfo.buffer.destroy();
        }
        console.log(`Destroyed ${this.buffers.size} buffers, freed ${this.formatBytes(this.totalAllocated)}`);
        this.buffers.clear();
        this.totalAllocated = 0;
    }
    getMemoryStats() {
        const bufferList = Array.from(this.buffers.entries()).map(([name, info]) => ({
            name,
            size: info.size,
            sizeFormatted: this.formatBytes(info.size),
            usage: this.formatUsage(info.usage),
            age: Date.now() - info.created
        }));

        return {
            totalBuffers: this.buffers.size,
            totalAllocated: this.totalAllocated,
            totalAllocatedFormatted: this.formatBytes(this.totalAllocated),
            buffers: bufferList
        };
    }
    async validateBuffer(bufferName, reference, tolerance = 1e-6) {
        const data = await this.readBuffer(bufferName);

        if (data.length !== reference.length) {
            return {
                passed: false,
                message: `Length mismatch: got ${data.length}, expected ${reference.length}`,
                maxError: Infinity,
                meanError: Infinity
            };
        }

        let maxError = 0;
        let totalError = 0;
        let errorCount = 0;

        for (let i = 0; i < data.length; i++) {
            const error = Math.abs(data[i] - reference[i]);
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
            tolerance,
            message: passed
                ? `Validation passed (max error: ${maxError.toExponential(3)})`
                : `Validation failed: ${errorCount} values exceed tolerance ${tolerance}`
        };
    }
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    formatUsage(usage) {
        const flags = [];
        if (usage & GPUBufferUsage.MAP_READ) flags.push('MAP_READ');
        if (usage & GPUBufferUsage.MAP_WRITE) flags.push('MAP_WRITE');
        if (usage & GPUBufferUsage.COPY_SRC) flags.push('COPY_SRC');
        if (usage & GPUBufferUsage.COPY_DST) flags.push('COPY_DST');
        if (usage & GPUBufferUsage.INDEX) flags.push('INDEX');
        if (usage & GPUBufferUsage.VERTEX) flags.push('VERTEX');
        if (usage & GPUBufferUsage.UNIFORM) flags.push('UNIFORM');
        if (usage & GPUBufferUsage.STORAGE) flags.push('STORAGE');
        if (usage & GPUBufferUsage.INDIRECT) flags.push('INDIRECT');
        if (usage & GPUBufferUsage.QUERY_RESOLVE) flags.push('QUERY_RESOLVE');
        return flags.join(' | ') || 'NONE';
    }
    logMemoryStats() {
        const stats = this.getMemoryStats();
        console.log('BufferManager Memory Statistics:');
        console.log(`Total buffers: ${stats.totalBuffers}`);
        console.log(`Total allocated: ${stats.totalAllocatedFormatted}`);

        if (stats.buffers.length > 0) {
            console.table(stats.buffers);
        }
    }
}
