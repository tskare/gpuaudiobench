import { GPUABenchmark } from '../core/GPUABenchmark.js';
import { BufferManager } from '../core/BufferManager.js';
import { VALIDATION_TOLERANCE } from '../core/ValidationConstants.js';

const MAX_GRID_DIMENSION = 48; // clamp to keep workloads reasonable in the browser

export class FDTD3DBenchmark extends GPUABenchmark {
    // NOTE: WebGPU defaults are intentionally smaller than STANDARD_PARAMS.md
    // (bufferSize=512, trackCount=128, grid=64x64x64) to ensure reasonable
    // performance in browser environments. FDTD is extremely memory and compute
    // intensive - full spec parameters can cause browser unresponsiveness.
    constructor(device, bufferSize = 256, trackCount = 4, options = {}) {
        super(device, 'FDTD3D', bufferSize, trackCount);
        this.bufferManager = new BufferManager(device);

        this.gridSize = {
            nx: this.clampDimension(Math.floor(options?.roomSizeX) || 32),
            ny: this.clampDimension(Math.floor(options?.roomSizeY) || 32),
            nz: this.clampDimension(Math.floor(options?.roomSizeZ) || 32)
        };

        this.absorption = Number(options?.absorptionCoeff) ?? 0.2;
        this.soundSpeed = Number(options?.soundSpeed) ?? 343;
        this.spatialStep = Number(options?.spatialStep) ?? 0.01;
        this.timeStep = this.computeStableTimeStep();

        this.sourcePosition = {
            x: Math.floor(this.gridSize.nx / 2),
            y: Math.floor(this.gridSize.ny / 2),
            z: Math.floor(this.gridSize.nz / 2)
        };

        this.receiverPositions = this.generateReceiverPositions();

        this.pipelines = null;
        this.bindGroups = null;

        this.pressureCPU = null;
        this.velocityXCPU = null;
        this.velocityYCPU = null;
        this.velocityZCPU = null;
        this.referenceOutput = null;
        this.sampleIndexSourceBuffer = null;
    }

    clampDimension(value) {
        return Math.max(8, Math.min(MAX_GRID_DIMENSION, value));
    }

    computeStableTimeStep() {
        const courant = 1 / Math.sqrt(3);
        const maxStep = this.spatialStep / (this.soundSpeed * Math.sqrt(3));
        return maxStep * 0.9 * courant;
    }

    generateReceiverPositions() {
        const receivers = [];
        const span = this.trackCount;
        for (let i = 0; i < this.trackCount; i++) {
            const ratio = span > 1 ? i / (span - 1) : 0.5;
            receivers.push({
                x: Math.floor(ratio * (this.gridSize.nx - 1)),
                y: Math.floor(this.gridSize.ny / 2),
                z: Math.floor(this.gridSize.nz / 2)
            });
        }
        return receivers;
    }
    async setup() {
        if (this.isSetup) {
            return;
        }

        this.shaderModule = await this.loadShader();
        await this.createPipelines();
        await this.setupBuffers();
        this.isSetup = true;
    }

    async loadShader() {
        return this.loadShaderFromFile('js/shaders/fdtd3d.wgsl');
    }

    async createPipelines() {
        this.velocityPipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'update_velocity'
            }
        });

        this.pressurePipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'update_pressure'
            }
        });

        this.injectPipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'inject_source'
            }
        });

        this.extractPipeline = await this.device.createComputePipelineAsync({
            layout: 'auto',
            compute: {
                module: this.shaderModule,
                entryPoint: 'extract_output'
            }
        });
    }

    async setupBuffers() {
        const { nx, ny, nz } = this.gridSize;
        const pressureSize = nx * ny * nz;
        const vxSize = (nx + 1) * ny * nz;
        const vySize = nx * (ny + 1) * nz;
        const vzSize = nx * ny * (nz + 1);

        this.pressureBuffer = this.createBuffer(
            'pressure',
            pressureSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );

        this.velocityXBuffer = this.createBuffer(
            'velocity_x',
            vxSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.velocityYBuffer = this.createBuffer(
            'velocity_y',
            vySize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.velocityZBuffer = this.createBuffer(
            'velocity_z',
            vzSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );

        this.inputAudio = new Float32Array(this.trackCount * this.bufferSize);
        for (let track = 0; track < this.trackCount; track++) {
            for (let i = 0; i < this.bufferSize; i++) {
                const idx = track * this.bufferSize + i;
                this.inputAudio[idx] = (Math.random() - 0.5) * 0.2;
            }
        }
        this.inputBuffer = this.createBuffer(
            'input_audio',
            this.inputAudio.length * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('input_audio', this.inputAudio);

        this.outputBuffer = this.createBuffer(
            'output',
            this.trackCount * this.bufferSize * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('output', new Float32Array(this.trackCount * this.bufferSize));

        const receiverData = new Uint32Array(this.trackCount * 4);
        this.receiverPositions.forEach((pos, index) => {
            const base = index * 4;
            receiverData[base + 0] = pos.x;
            receiverData[base + 1] = pos.y;
            receiverData[base + 2] = pos.z;
            receiverData[base + 3] = 0;
        });
        this.receiverBuffer = this.createBuffer(
            'receivers',
            receiverData.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.receiverBuffer, 0, receiverData);

        this.sampleIndexBuffer = this.createBuffer(
            'sample_index',
            4,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        const sampleIndexData = new Uint32Array(this.bufferSize);
        for (let i = 0; i < this.bufferSize; i++) {
            sampleIndexData[i] = i;
        }
        this.sampleIndexSourceBuffer = this.createBuffer(
            'sample_index_source',
            sampleIndexData.byteLength,
            GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        );
        this.writeBuffer('sample_index_source', sampleIndexData);

        const paramsData = new ArrayBuffer(64);
        const paramsView = new DataView(paramsData);
        paramsView.setUint32(0, nx, true);
        paramsView.setUint32(4, ny, true);
        paramsView.setUint32(8, nz, true);
        paramsView.setUint32(12, this.trackCount, true);
        paramsView.setFloat32(16, this.soundSpeed, true);
        paramsView.setFloat32(20, this.spatialStep, true);
        paramsView.setFloat32(24, this.timeStep, true);
        paramsView.setFloat32(28, this.absorption, true);
        paramsView.setUint32(32, this.sourcePosition.x, true);
        paramsView.setUint32(36, this.sourcePosition.y, true);
        paramsView.setUint32(40, this.sourcePosition.z, true);
        paramsView.setUint32(44, this.bufferSize, true);
        // padding for 64 bytes total

        this.paramsBuffer = this.createBuffer(
            'params',
            paramsData.byteLength,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        );
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

        this.createBindGroups();
        this.generateCPUReference();
    }

    createBindGroups() {
        this.velocityBindGroup = this.device.createBindGroup({
            layout: this.velocityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.pressureBuffer } },
                { binding: 1, resource: { buffer: this.velocityXBuffer } },
                { binding: 2, resource: { buffer: this.velocityYBuffer } },
                { binding: 3, resource: { buffer: this.velocityZBuffer } },
                { binding: 4, resource: { buffer: this.paramsBuffer } }
            ]
        });

        this.pressureBindGroup = this.device.createBindGroup({
            layout: this.pressurePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 5, resource: { buffer: this.pressureBuffer } },
                { binding: 6, resource: { buffer: this.velocityXBuffer } },
                { binding: 7, resource: { buffer: this.velocityYBuffer } },
                { binding: 8, resource: { buffer: this.velocityZBuffer } },
                { binding: 9, resource: { buffer: this.paramsBuffer } }
            ]
        });

        this.injectBindGroup = this.device.createBindGroup({
            layout: this.injectPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 10, resource: { buffer: this.pressureBuffer } },
                { binding: 11, resource: { buffer: this.inputBuffer } },
                { binding: 12, resource: { buffer: this.paramsBuffer } },
                { binding: 13, resource: { buffer: this.sampleIndexBuffer } }
            ]
        });

        this.extractBindGroup = this.device.createBindGroup({
            layout: this.extractPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 14, resource: { buffer: this.pressureBuffer } },
                { binding: 15, resource: { buffer: this.outputBuffer } },
                { binding: 16, resource: { buffer: this.paramsBuffer } },
                { binding: 17, resource: { buffer: this.sampleIndexBuffer } },
                { binding: 18, resource: { buffer: this.receiverBuffer } }
            ]
        });
    }

    generateCPUReference() {
        const { nx, ny, nz } = this.gridSize;
        const pressure = new Float32Array(nx * ny * nz);
        const vx = new Float32Array((nx + 1) * ny * nz);
        const vy = new Float32Array(nx * (ny + 1) * nz);
        const vz = new Float32Array(nx * ny * (nz + 1));
        const output = new Float32Array(this.trackCount * this.bufferSize);

        const index3D = (x, y, z) => z * nx * ny + y * nx + x;
        const vxIndex = (x, y, z) => z * (nx + 1) * ny + y * (nx + 1) + x;
        const vyIndex = (x, y, z) => z * nx * (ny + 1) + y * nx + x;
        const vzIndex = (x, y, z) => z * nx * ny + y * nx + x;

        const rho0 = 1.225;
        const dt_over_rho_dx = this.timeStep / (rho0 * this.spatialStep);
        const rho_c2_dt_over_dx = rho0 * this.soundSpeed * this.soundSpeed * this.timeStep / this.spatialStep;

        const sourceIndex = index3D(this.sourcePosition.x, this.sourcePosition.y, this.sourcePosition.z);
        const receiverIndices = this.receiverPositions.map(pos => index3D(pos.x, pos.y, pos.z));

        for (let step = 0; step < this.bufferSize; step++) {
            for (let z = 0; z < nz; z++) {
                for (let y = 0; y < ny; y++) {
                    for (let x = 1; x < nx; x++) {
                        const idx = vxIndex(x, y, z);
                        const pRight = pressure[index3D(x, y, z)];
                        const pLeft = pressure[index3D(x - 1, y, z)];
                        vx[idx] -= dt_over_rho_dx * (pRight - pLeft);
                    }
                }
            }

            for (let z = 0; z < nz; z++) {
                for (let y = 1; y < ny; y++) {
                    for (let x = 0; x < nx; x++) {
                        const idx = vyIndex(x, y, z);
                        const pBack = pressure[index3D(x, y, z)];
                        const pFront = pressure[index3D(x, y - 1, z)];
                        vy[idx] -= dt_over_rho_dx * (pBack - pFront);
                    }
                }
            }

            for (let z = 1; z < nz; z++) {
                for (let y = 0; y < ny; y++) {
                    for (let x = 0; x < nx; x++) {
                        const idx = vzIndex(x, y, z);
                        const pTop = pressure[index3D(x, y, z)];
                        const pBottom = pressure[index3D(x, y, z - 1)];
                        vz[idx] -= dt_over_rho_dx * (pTop - pBottom);
                    }
                }
            }

            for (let z = 1; z < nz - 1; z++) {
                for (let y = 1; y < ny - 1; y++) {
                    for (let x = 1; x < nx - 1; x++) {
                        const idx = index3D(x, y, z);
                        const vxRight = vx[vxIndex(x + 1, y, z)];
                        const vxLeft = vx[vxIndex(x, y, z)];
                        const vyBack = vy[vyIndex(x, y + 1, z)];
                        const vyFront = vy[vyIndex(x, y, z)];
                        const vzTop = vz[vzIndex(x, y, z + 1)];
                        const vzBottom = vz[vzIndex(x, y, z)];

                        const divergence = (vxRight - vxLeft) + (vyBack - vyFront) + (vzTop - vzBottom);
                        pressure[idx] -= rho_c2_dt_over_dx * divergence;
                    }
                }
            }

            for (let z = 0; z < nz; z++) {
                for (let y = 0; y < ny; y++) {
                    pressure[index3D(0, y, z)] *= (1 - this.absorption);
                    pressure[index3D(nx - 1, y, z)] *= (1 - this.absorption);
                }
            }
            for (let z = 0; z < nz; z++) {
                for (let x = 0; x < nx; x++) {
                    pressure[index3D(x, 0, z)] *= (1 - this.absorption);
                    pressure[index3D(x, ny - 1, z)] *= (1 - this.absorption);
                }
            }
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    pressure[index3D(x, y, 0)] *= (1 - this.absorption);
                    pressure[index3D(x, y, nz - 1)] *= (1 - this.absorption);
                }
            }

            const sourceSample = this.inputAudio[step];
            pressure[sourceIndex] += sourceSample * 0.1;

            for (let track = 0; track < this.trackCount; track++) {
                output[track * this.bufferSize + step] = pressure[receiverIndices[track]];
            }
        }

        this.pressureCPU = pressure;
        this.velocityXCPU = vx;
        this.velocityYCPU = vy;
        this.velocityZCPU = vz;
        this.referenceOutput = output;
    }

    resetStateBuffers() {
        const zeroPressure = this.getZeroedArray('f32', this.gridSize.nx * this.gridSize.ny * this.gridSize.nz);
        const zeroVx = this.getZeroedArray('f32', (this.gridSize.nx + 1) * this.gridSize.ny * this.gridSize.nz);
        const zeroVy = this.getZeroedArray('f32', this.gridSize.nx * (this.gridSize.ny + 1) * this.gridSize.nz);
        const zeroVz = this.getZeroedArray('f32', this.gridSize.nx * this.gridSize.ny * (this.gridSize.nz + 1));
        const zeroOutput = this.getZeroedArray('f32', this.trackCount * this.bufferSize);
        this.writeBuffer('pressure', zeroPressure);
        this.writeBuffer('velocity_x', zeroVx);
        this.writeBuffer('velocity_y', zeroVy);
        this.writeBuffer('velocity_z', zeroVz);
        this.writeBuffer('output', zeroOutput);
    }

    async performIteration() {
        this.resetStateBuffers();

        const { nx, ny, nz } = this.gridSize;
        const workgroupSize = 4;
        const workgroupsVelocity = [
            Math.ceil(nx / workgroupSize),
            Math.ceil(ny / workgroupSize),
            Math.ceil(nz / workgroupSize)
        ];
        const workgroupsPressure = [
            Math.ceil(nx / workgroupSize),
            Math.ceil(ny / workgroupSize),
            Math.ceil(nz / workgroupSize)
        ];

        const commandBuffers = [];
        const batchSize = 8;

        for (let step = 0; step < this.bufferSize; step++) {
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(
                this.sampleIndexSourceBuffer,
                step * 4,
                this.sampleIndexBuffer,
                0,
                4
            );

            {
                const pass = encoder.beginComputePass();
                pass.setPipeline(this.velocityPipeline);
                pass.setBindGroup(0, this.velocityBindGroup);
                pass.dispatchWorkgroups(workgroupsVelocity[0], workgroupsVelocity[1], workgroupsVelocity[2]);
                pass.end();
            }

            {
                const pass = encoder.beginComputePass();
                pass.setPipeline(this.pressurePipeline);
                pass.setBindGroup(0, this.pressureBindGroup);
                pass.dispatchWorkgroups(workgroupsPressure[0], workgroupsPressure[1], workgroupsPressure[2]);
                pass.end();
            }

            {
                const pass = encoder.beginComputePass();
                pass.setPipeline(this.injectPipeline);
                pass.setBindGroup(0, this.injectBindGroup);
                pass.dispatchWorkgroups(this.trackCount);
                pass.end();
            }

            {
                const pass = encoder.beginComputePass();
                pass.setPipeline(this.extractPipeline);
                pass.setBindGroup(0, this.extractBindGroup);
                pass.dispatchWorkgroups(this.trackCount);
                pass.end();
            }

            commandBuffers.push(encoder.finish());

            if (commandBuffers.length >= batchSize) {
                this.device.queue.submit(commandBuffers);
                commandBuffers.length = 0;
            }
        }

        if (commandBuffers.length > 0) {
            this.device.queue.submit(commandBuffers);
        }

        await this.device.queue.onSubmittedWorkDone();
    }

    async validate() {
        const gpuOutput = await this.readBuffer('output');
        return this.validateOutput(gpuOutput, this.referenceOutput, VALIDATION_TOLERANCE.FDTD_SIMULATION);
    }

    getMetadata() {
        return {
            ...super.getMetadata?.() || {},
            benchmark_type: 'fdtd3d',
            description: '3D finite difference time domain acoustics',
            grid_dimensions: this.gridSize,
            sound_speed: this.soundSpeed,
            spatial_step: this.spatialStep,
            time_step: this.timeStep,
            absorption: this.absorption,
            source_position: this.sourcePosition,
            receiver_positions: this.receiverPositions,
            workgroup_size: 64,
            workgroups_velocity: {
                x: Math.ceil(this.gridSize.nx / 4),
                y: Math.ceil(this.gridSize.ny / 4),
                z: Math.ceil(this.gridSize.nz / 4)
            },
            total_samples: this.bufferSize * this.trackCount
        };
    }

    cleanup() {
        super.cleanup();
        this.bufferManager.destroyAll();
        this.pressureCPU = null;
        this.velocityXCPU = null;
        this.velocityYCPU = null;
        this.velocityZCPU = null;
        this.referenceOutput = null;
    }
}
