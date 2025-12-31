import { NoOpBenchmark } from './benchmarks/NoOpBenchmark.js';
import { GainBenchmark } from './benchmarks/GainBenchmark.js';
import { GainStatsBenchmark } from './benchmarks/GainStatsBenchmark.js';
import { Convolution1DBenchmark } from './benchmarks/Convolution1DBenchmark.js';
import { Convolution1DAccelBenchmark } from './benchmarks/Convolution1DAccelBenchmark.js';
import { DataTransferBenchmark } from './benchmarks/DataTransferBenchmark.js';
import { IIRFilterBenchmark } from './benchmarks/IIRFilterBenchmark.js';
import { RandomMemoryBenchmark } from './benchmarks/RandomMemoryBenchmark.js';
import { FFTBenchmark } from './benchmarks/FFTBenchmark.js';
import { ModalFilterBankBenchmark } from './benchmarks/ModalFilterBankBenchmark.js';
import { DWG1DNaiveBenchmark } from './benchmarks/DWG1DNaiveBenchmark.js';
import { DWG1DAccelBenchmark } from './benchmarks/DWG1DAccelBenchmark.js';
import { FDTD3DBenchmark } from './benchmarks/FDTD3DBenchmark.js';
import { Statistics } from './core/Statistics.js';
import { ParameterBuilder } from './core/ParameterBuilder.js';

const CONSTANTS = {
    SUITE_MAX_ITERATIONS: 25,

    DEFAULT_CHART_BINS: 30,
    BYTES_PER_FLOAT: 4,
    MS_TO_SECONDS: 1000,
    BYTES_TO_GB: 1024 * 1024 * 1024
};

class WebGPUBenchmarkApp {
    constructor() {
        this.device = null;
        this.adapter = null;
        this.currentBenchmark = null;
        this.charts = new Map();
        this.isInitialized = false;
        this.selectedBenchmark = null;
        this.benchmarkCategories = this.initializeBenchmarkCategories();
        this.isRunningSuite = false;
        this.suiteResults = new Map();
        this.benchmarkParameters = new Map();
        this.benchmarkFactory = this.initializeBenchmarkFactory();
        this.quietMode = false;
    }
    log(...args) {
        if (!this.quietMode) {
            console.log(...args);
        }
    }

    initializeBenchmarkCategories() {
        return {
            basic: [
                { id: 'noop', name: 'NoOp', description: 'Kernel launch overhead measurement - baseline cost of launching GPU kernels without computation.', icon: '⚡' },
                { id: 'gain', name: 'Gain', description: 'Gain processing - multiplies audio samples by a fixed gain value.', icon: '🎚️' },
                { id: 'gainstats', name: 'GainStats', description: 'Gain processing with statistical analysis - min/max/mean calculation.', icon: '📊' }
            ],
            datatransfer: [
                { id: 'datacopy0199', name: 'DataCopy 01/99', description: 'Data transfer with 1% input, 99% output ratio - output-heavy scenarios.', icon: '📤' },
                { id: 'datacopy2080', name: 'DataCopy 20/80', description: 'Data transfer with 20% input, 80% output ratio - output-biased processing.', icon: '📊' },
                { id: 'datacopy5050', name: 'DataCopy 50/50', description: 'Data transfer with 50% input, 50% output ratio - balanced I/O.', icon: '⚖️' },
                { id: 'datacopy8020', name: 'DataCopy 80/20', description: 'Data transfer with 80% input, 20% output ratio - input-heavy scenarios.', icon: '📥' },
                { id: 'datacopy9901', name: 'DataCopy 99/01', description: 'Data transfer with 99% input, 1% output ratio - input-dominant scenarios.', icon: '📨' }
            ],
            dsp: [
                { id: 'iirfilter', name: 'IIR Filter', description: '2nd order biquad digital filter (Direct Form II) - Butterworth lowpass at fs/4, Q=0.707.', icon: '🎛️' },
                { id: 'conv1d', name: 'Conv1D', description: '1D convolution - constant/texture memory vs device memory for reverb and filtering.', icon: '〰️' },
                { id: 'conv1daccel', name: 'Conv1D Accel', description: 'Sample-parallel FIR convolution with shared-memory tiling.', icon: '🚀' },
                { id: 'modalfilterbank', name: 'Modal Filter Bank', description: 'Modal synthesis filter bank - complex arithmetic with up to 1M modes.', icon: '🔊' },
                { id: 'fft', name: 'FFT', description: 'Fast Fourier Transform - Cooley-Tukey radix-2 implementation for frequency domain processing.', icon: '📈' }
            ],
            synthesis: [
                { id: 'dwg1dnaive', name: 'DWG1D Naive', description: 'Digital Waveguide - straightforward GPU parallelization for string synthesis.', icon: '🎸' },
                { id: 'dwg1daccel', name: 'DWG1D Accel', description: 'Digital Waveguide - optimized implementation for string synthesis.', icon: '🎯' },
                { id: 'fdtd3d', name: 'FDTD3D', description: '3D Finite Difference Time Domain - acoustic wave propagation for room acoustics.', icon: '🌊' }
            ],
            memory: [
                { id: 'randommemory', name: 'Random Memory', description: 'Random memory access - non-coalesced memory patterns for granular synthesis simulation.', icon: '🎲' }
            ]
        };
    }
    initializeBenchmarkFactory() {
        return {
            'noop': (device, config) => new NoOpBenchmark(device, config.bufferSize, config.trackCount),
            'gain': (device, config) => new GainBenchmark(device, config.bufferSize, config.trackCount),
            'gainstats': (device, config) => new GainStatsBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'conv1d': (device, config) => new Convolution1DBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'conv1daccel': (device, config) => new Convolution1DAccelBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'iirfilter': (device, config) => new IIRFilterBenchmark(device, config.bufferSize, config.trackCount),
            'fft': (device, config) => new FFTBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'modalfilterbank': (device, config) => new ModalFilterBankBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'dwg1dnaive': (device, config) => new DWG1DNaiveBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'dwg1daccel': (device, config) => new DWG1DAccelBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'fdtd3d': (device, config) => new FDTD3DBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'randommemory': (device, config) => new RandomMemoryBenchmark(device, config.bufferSize, config.trackCount, config.parameters || {}),
            'datacopy0199': (device, config) => new DataTransferBenchmark(device, 0.01, 0.99, config.bufferSize, config.trackCount),
            'datacopy2080': (device, config) => new DataTransferBenchmark(device, 0.20, 0.80, config.bufferSize, config.trackCount),
            'datacopy5050': (device, config) => new DataTransferBenchmark(device, 0.50, 0.50, config.bufferSize, config.trackCount),
            'datacopy8020': (device, config) => new DataTransferBenchmark(device, 0.80, 0.20, config.bufferSize, config.trackCount),
            'datacopy9901': (device, config) => new DataTransferBenchmark(device, 0.99, 0.01, config.bufferSize, config.trackCount)
        };
    }
    async initialize() {
        try {
            console.log('Initializing WebGPU Audio Benchmark Suite...');

            if (!navigator.gpu) {
                this.showWebGPUError('WebGPU is not supported in this browser');
                return false;
            }

            this.adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });

            if (!this.adapter) {
                this.showWebGPUError('No WebGPU adapter found');
                return false;
            }

            this.device = await this.adapter.requestDevice();

            if (!this.device) {
                this.showWebGPUError('Failed to create WebGPU device');
                return false;
            }

            this.device.lost.then((info) => {
                console.error('WebGPU device lost:', info);
                const reason = info?.message || info?.reason || 'Unknown reason';
                this.showWebGPUError(
                    `WebGPU device lost: ${reason}. Reload the page and close other GPU-heavy tabs if the issue persists.`
                );
            });

            console.log('WebGPU initialized successfully');
            console.log('Adapter info:', {
                vendor: this.adapter.info?.vendor || 'Unknown',
                architecture: this.adapter.info?.architecture || 'Unknown',
                device: this.adapter.info?.device || 'Unknown'
            });

            this.isInitialized = true;
            this.setupEventListeners();
            this.setupBenchmarkUI();
            this.setupThemeToggle();
            this.populateHardwareInfo();
            this.hideWebGPUError();

            return true;
        } catch (error) {
            console.error('Failed to initialize WebGPU:', error);
            this.showWebGPUError(`WebGPU initialization failed: ${error.message}`);
            return false;
        }
    }
    showWebGPUError(message) {
        const alert = document.getElementById('webgpu-alert');
        const alertContent = alert.querySelector('strong').nextSibling;
        alertContent.textContent = ` ${message}`;
        alert.style.display = 'block';
    }
    hideWebGPUError() {
        const alert = document.getElementById('webgpu-alert');
        alert.style.display = 'none';
    }
    setupBenchmarkUI() {
        const categorySelect = document.getElementById('benchmark-category');
        categorySelect.addEventListener('sl-change', (event) => {
            this.renderBenchmarkCards(event.target.value);
            this.updateRunAllButton(event.target.value);
        });

        const runAllButton = document.getElementById('run-all-category');
        runAllButton.addEventListener('click', () => {
            const category = categorySelect.value;
            this.runBenchmarkSuite(category);
        });

        this.renderBenchmarkCards('basic');
        this.updateRunAllButton('basic');
    }
    updateRunAllButton(category) {
        const runAllButton = document.getElementById('run-all-category');
        const benchmarks = this.benchmarkCategories[category] || [];

        runAllButton.disabled = benchmarks.length === 0 || this.isRunningSuite;

        const categoryName = document.querySelector(`sl-option[value="${category}"]`)?.textContent || category;

        while (runAllButton.firstChild) {
            runAllButton.removeChild(runAllButton.firstChild);
        }

        const icon = document.createElement('sl-icon');
        icon.slot = 'prefix';
        icon.name = 'play-circle';
        runAllButton.appendChild(icon);

        const textNode = document.createTextNode(`Run All ${categoryName} (${benchmarks.length})`);
        runAllButton.appendChild(textNode);
    }
    async runBenchmarkSuite(category) {
        if (this.isRunningSuite) return;

        const benchmarks = this.benchmarkCategories[category] || [];
        if (benchmarks.length === 0) return;

        this.isRunningSuite = true;
        this.suiteResults.clear();

        const progressContainer = document.getElementById('suite-progress');
        const progressBar = document.getElementById('suite-progress-bar');
        const statusText = document.getElementById('suite-status');

        progressContainer.style.display = 'block';
        progressBar.value = 0;

        document.getElementById('run-all-category').disabled = true;
        document.getElementById('benchmark-category').disabled = true;

        try {
            for (let i = 0; i < benchmarks.length; i++) {
                const benchmark = benchmarks[i];

                statusText.textContent = `Running ${benchmark.name} (${i + 1}/${benchmarks.length})...`;

                this.selectBenchmark(benchmark.id);

                await this.runBenchmarkForSuite(benchmark.id);

                progressBar.value = ((i + 1) / benchmarks.length) * 100;

                await new Promise(resolve => setTimeout(resolve, 500));
            }

            statusText.textContent = `Completed all ${benchmarks.length} benchmarks in category!`;

            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 2000);

        } catch (error) {
            console.error('Suite run failed:', error);
            statusText.textContent = 'Suite run failed. Check console for details.';
        } finally {
            this.isRunningSuite = false;

            document.getElementById('run-all-category').disabled = false;
            document.getElementById('benchmark-category').disabled = false;
            this.updateRunAllButton(category);
        }
    }
    async runBenchmarkForSuite(benchmarkId) {
        try {
            const config = this.getConfiguration(benchmarkId);
            this.log(`Running benchmark: ${benchmarkId}`);

            config.iterations = Math.min(config.iterations, CONSTANTS.SUITE_MAX_ITERATIONS);

            const { benchmark, results } = await this._executeBenchmark(benchmarkId, config);

            this.suiteResults.set(benchmarkId, {
                median: results.statistics.median,
                p95: results.statistics.p95,
                max: results.statistics.max,
                min: results.statistics.min,
                validation: results.validation.passed ? 'PASSED' : 'FAILED',
                timestamp: new Date().toISOString()
            });

            if (benchmark && typeof benchmark.cleanup === 'function') {
                try {
                    benchmark.cleanup();
                } catch (cleanupError) {
                    console.warn('Error during benchmark cleanup:', cleanupError);
                }
            }

        } catch (error) {
            console.error(`Failed to run ${benchmarkId}:`, error);
            this.suiteResults.set(benchmarkId, {
                error: error.message,
                timestamp: new Date().toISOString()
            });
        }
    }
    renderBenchmarkCards(category) {
        const cardsContainer = document.getElementById('benchmark-cards');
        const benchmarks = this.benchmarkCategories[category] || [];

        while (cardsContainer.firstChild) {
            cardsContainer.removeChild(cardsContainer.firstChild);
        }

        benchmarks.forEach(benchmark => {
            const card = document.createElement('div');
            card.className = 'benchmark-card';
            card.dataset.benchmarkId = benchmark.id;

            const iconDiv = document.createElement('div');
            iconDiv.className = 'benchmark-icon';
            iconDiv.textContent = benchmark.icon;
            card.appendChild(iconDiv);

            const title = document.createElement('h4');
            title.textContent = benchmark.name;
            card.appendChild(title);

            const description = document.createElement('p');
            description.textContent = benchmark.description;
            card.appendChild(description);

            card.addEventListener('click', () => {
                this.selectBenchmark(benchmark.id);
            });

            cardsContainer.appendChild(card);
        });
    }
    selectBenchmark(benchmarkId) {
        document.querySelectorAll('.benchmark-card').forEach(card => {
            card.classList.remove('selected');
        });

        const selectedCard = document.querySelector(`[data-benchmark-id="${benchmarkId}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
        }

        this.selectedBenchmark = benchmarkId;
        this.showBenchmarkContent(benchmarkId);
    }
    createElement(tag, attributes = {}, children = []) {
        const element = document.createElement(tag);

        Object.entries(attributes).forEach(([key, value]) => {
            if (key === 'textContent') {
                element.textContent = value;
            } else if (key === 'className') {
                element.className = value;
            } else if (key === 'style' && typeof value === 'object') {
                Object.assign(element.style, value);
            } else {
                element.setAttribute(key, value);
            }
        });

        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else if (child) {
                element.appendChild(child);
            }
        });

        return element;
    }
    showBenchmarkContent(benchmarkId) {
        const contentArea = document.getElementById('benchmark-content');

        while (contentArea.firstChild) {
            contentArea.removeChild(contentArea.firstChild);
        }

        const title = this.createElement('h3', { textContent: `${this.getBenchmarkName(benchmarkId)} Benchmark` });
        contentArea.appendChild(title);

        const description = this.createElement('p', { textContent: this.getBenchmarkDescription(benchmarkId) });
        contentArea.appendChild(description);

        const controlsDiv = this.createElement('div', { className: 'benchmark-controls' });

        const runButton = this.createElement('sl-button', {
            id: `run-${benchmarkId}`,
            variant: 'primary',
            size: 'large'
        }, [
            this.createElement('sl-icon', { slot: 'prefix', name: 'play-fill' }),
            `Run ${this.getBenchmarkName(benchmarkId)} Benchmark`
        ]);
        controlsDiv.appendChild(runButton);

        const exportButton = this.createElement('sl-button', {
            id: `export-${benchmarkId}`,
            variant: 'default',
            disabled: 'disabled'
        }, [
            this.createElement('sl-icon', { slot: 'prefix', name: 'download' }),
            'Export Results'
        ]);
        controlsDiv.appendChild(exportButton);

        contentArea.appendChild(controlsDiv);

        const parameterControlsDOM = this.createParameterControlsDOM(benchmarkId);
        if (parameterControlsDOM) {
            contentArea.appendChild(parameterControlsDOM);
        }

        const resultsDiv = this.createElement('div', {
            id: `${benchmarkId}-results`,
            className: 'results-section',
            style: { display: 'none' }
        });

        const resultsCard = this.createElement('sl-card');

        const headerDiv = this.createElement('div', { slot: 'header', textContent: 'Results' });
        resultsCard.appendChild(headerDiv);

        const statsGrid = this.createElement('div', { className: 'stats-grid' });

        const statItems = [
            { label: 'Median:', id: `${benchmarkId}-median` },
            { label: 'P95:', id: `${benchmarkId}-p95` },
            { label: 'Max:', id: `${benchmarkId}-max` },
            { label: 'Min:', id: `${benchmarkId}-min` }
        ];

        statItems.forEach(stat => {
            const statDiv = this.createElement('div', { className: 'stat' }, [
                this.createElement('span', { className: 'stat-label', textContent: stat.label }),
                this.createElement('span', { id: stat.id, className: 'stat-value', textContent: '-' })
            ]);
            statsGrid.appendChild(statDiv);
        });

        resultsCard.appendChild(statsGrid);

        const canvas = this.createElement('canvas', {
            id: `${benchmarkId}-chart`,
            width: '600',
            height: '300'
        });
        resultsCard.appendChild(canvas);

        resultsDiv.appendChild(resultsCard);
        contentArea.appendChild(resultsDiv);

        contentArea.style.display = 'block';

        document.getElementById(`run-${benchmarkId}`).addEventListener('click', () => {
            this.runBenchmark(benchmarkId);
        });

        document.getElementById(`export-${benchmarkId}`).addEventListener('click', () => {
            this.exportResults(benchmarkId);
        });

        const params = this.getBenchmarkParameters(benchmarkId);
        if (params.length > 0) {
            const existingParams = this.benchmarkParameters.get(benchmarkId);
            if (existingParams && existingParams.size > 0) {
                this.applyParametersToUI(benchmarkId, existingParams);
            } else if (this.hasStoredParameters(benchmarkId)) {
                this.loadParametersFromStorage(benchmarkId);
            } else {
                this.resetParametersToDefaults(benchmarkId);
            }
            this.setupParameterListeners(benchmarkId);
        }
    }
    getBenchmarkName(benchmarkId) {
        for (const category of Object.values(this.benchmarkCategories)) {
            const benchmark = category.find(b => b.id === benchmarkId);
            if (benchmark) return benchmark.name;
        }
        return benchmarkId;
    }
    getBenchmarkDescription(benchmarkId) {
        for (const category of Object.values(this.benchmarkCategories)) {
            const benchmark = category.find(b => b.id === benchmarkId);
            if (benchmark) return benchmark.description;
        }
        return 'No description available.';
    }
    setupThemeToggle() {
        const themeToggle = document.getElementById('theme-toggle');

        const savedTheme = localStorage.getItem('webgpu-benchmark-theme') || 'light';
        this.setTheme(savedTheme);

        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            this.setTheme(newTheme);
        });
    }
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('webgpu-benchmark-theme', theme);

        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.name = theme === 'light' ? 'moon' : 'sun';
            themeToggle.label = theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode';
        }
    }
    populateHardwareInfo() {
        if (this.adapter?.info) {
            document.getElementById('gpu-vendor').textContent = this.adapter.info.vendor || 'Unknown';
            document.getElementById('gpu-architecture').textContent = this.adapter.info.architecture || 'Unknown';
            document.getElementById('gpu-device').textContent = this.adapter.info.device || 'Unknown';
        }

        if (this.device?.limits) {
            const limits = this.device.limits;

            const formatBytes = (bytes) => {
                if (bytes >= 1024 * 1024 * 1024) {
                    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
                } else if (bytes >= 1024 * 1024) {
                    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
                } else if (bytes >= 1024) {
                    return `${(bytes / 1024).toFixed(1)} KB`;
                }
                return `${bytes} bytes`;
            };

            document.getElementById('max-buffer-size').textContent =
                limits.maxBufferSize ? formatBytes(limits.maxBufferSize) : 'Unknown';

            document.getElementById('max-workgroup-size').textContent =
                limits.maxComputeWorkgroupSizeX ? `${limits.maxComputeWorkgroupSizeX}×${limits.maxComputeWorkgroupSizeY}×${limits.maxComputeWorkgroupSizeZ}` : 'Unknown';

            document.getElementById('max-storage-buffer').textContent =
                limits.maxStorageBufferBindingSize ? formatBytes(limits.maxStorageBufferBindingSize) : 'Unknown';
        }

        const userAgent = navigator.userAgent;
        const browserInfo = this.getBrowserInfo(userAgent);
        document.getElementById('user-agent').textContent = browserInfo;
    }
    getBrowserInfo(userAgent) {
        if (userAgent.includes('Chrome')) {
            const match = userAgent.match(/Chrome\/(\d+)/);
            return match ? `Chrome ${match[1]}` : 'Chrome (unknown version)';
        } else if (userAgent.includes('Firefox')) {
            const match = userAgent.match(/Firefox\/(\d+)/);
            return match ? `Firefox ${match[1]}` : 'Firefox (unknown version)';
        } else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
            const match = userAgent.match(/Version\/(\d+)/);
            return match ? `Safari ${match[1]}` : 'Safari (unknown version)';
        } else if (userAgent.includes('Edge')) {
            const match = userAgent.match(/Edg\/(\d+)/);
            return match ? `Edge ${match[1]}` : 'Edge (unknown version)';
        }
        return 'Unknown browser';
    }
    getBenchmarkParameters(benchmarkId) {
        const paramConfigs = {
            gainstats: [
                ParameterBuilder.slider('gainValue', 'Gain Multiplier', 0.1, 6.0, 2.0, 0.1, '', 'Gain applied to each audio sample')
            ],
            iirfilter: ParameterBuilder.filterParams(),
            conv1d: ParameterBuilder.convolutionParams(),
            conv1daccel: ParameterBuilder.convolutionParams(),
            fft: ParameterBuilder.fftParams(),
            randommemory: ParameterBuilder.memoryParams(),
            modalfilterbank: ParameterBuilder.modalParams(),
            dwg1dnaive: ParameterBuilder.waveguideParams(100, null),
            dwg1daccel: ParameterBuilder.waveguideParams(100, true),
            fdtd3d: ParameterBuilder.fdtdParams()
        };

        return paramConfigs[benchmarkId] || [];
    }
    getBenchmarkParameterValues(benchmarkId) {
        const params = this.getBenchmarkParameters(benchmarkId);
        if (!params || params.length === 0) {
            return {};
        }

        const stored = this.benchmarkParameters.get(benchmarkId);
        const values = {};

        params.forEach(param => {
            let value = stored?.get(param.id);
            if (value === undefined) {
                value = param.default;
            }

            if (param.type === 'slider') {
                const numericValue = typeof value === 'number' ? value : parseFloat(value);
                values[param.id] = Number.isFinite(numericValue) ? numericValue : param.default;
            } else if (param.type === 'switch') {
                if (typeof value === 'boolean') {
                    values[param.id] = value;
                } else {
                    values[param.id] = value === 'true' || value === '1' || value === 1;
                }
            } else if (param.type === 'select') {
                if (typeof value === 'number') {
                    values[param.id] = value;
                } else {
                    const numericValue = Number(value);
                    values[param.id] = Number.isFinite(numericValue) ? numericValue : value;
                }
            } else {
                values[param.id] = value;
            }
        });

        return values;
    }
    createParameterControlsDOM(benchmarkId) {
        const params = this.getBenchmarkParameters(benchmarkId);
        if (!params || params.length === 0) return null;

        const details = this.createElement('sl-details', {
            summary: 'Advanced Parameters',
            className: 'advanced-params',
            open: 'true'
        });

        const parameterGrid = this.createElement('div', { className: 'parameter-grid' });

        params.forEach(param => {
            const control = this.createParameterControlDOM(benchmarkId, param);
            if (control) {
                parameterGrid.appendChild(control);
            }
        });

        details.appendChild(parameterGrid);

        const actionsDiv = this.createElement('div', { className: 'parameter-actions' });

        const resetButton = this.createElement('sl-button', {
            id: `reset-params-${benchmarkId}`,
            variant: 'outline',
            size: 'small'
        }, [
            this.createElement('sl-icon', { slot: 'prefix', name: 'arrow-clockwise' }),
            'Reset to Defaults'
        ]);
        actionsDiv.appendChild(resetButton);

        const loadButton = this.createElement('sl-button', {
            id: `load-preset-${benchmarkId}`,
            variant: 'outline',
            size: 'small'
        }, [
            this.createElement('sl-icon', { slot: 'prefix', name: 'folder-open' }),
            'Load Preset'
        ]);
        actionsDiv.appendChild(loadButton);

        const saveButton = this.createElement('sl-button', {
            id: `save-preset-${benchmarkId}`,
            variant: 'outline',
            size: 'small'
        }, [
            this.createElement('sl-icon', { slot: 'prefix', name: 'floppy' }),
            'Save Preset'
        ]);
        actionsDiv.appendChild(saveButton);

        details.appendChild(actionsDiv);

        return details;
    }
    createParameterControlDOM(benchmarkId, param) {
        const paramId = `${benchmarkId}-${param.id}`;
        const unit = param.unit ? ` ${param.unit}` : '';

        const controlDiv = this.createElement('div', { className: 'parameter-control' });

        switch (param.type) {
            case 'slider': {
                const range = this.createElement('sl-range', {
                    id: paramId,
                    label: param.label,
                    min: param.min.toString(),
                    max: param.max.toString(),
                    value: param.default.toString(),
                    step: param.step.toString(),
                    'help-text': param.description
                });
                const suffix = this.createElement('span', {
                    slot: 'suffix',
                    textContent: `${param.default}${unit}`
                });
                range.appendChild(suffix);
                controlDiv.appendChild(range);
                break;
            }

            case 'select': {
                const select = this.createElement('sl-select', {
                    id: paramId,
                    label: param.label,
                    value: param.default.toString(),
                    'help-text': param.description
                });
                param.options.forEach(opt => {
                    const option = this.createElement('sl-option', {
                        value: opt.value.toString(),
                        textContent: opt.label
                    });
                    select.appendChild(option);
                });
                controlDiv.appendChild(select);
                break;
            }

            case 'switch': {
                const switchEl = this.createElement('sl-switch', {
                    id: paramId,
                    'help-text': param.description
                }, [param.label]);
                if (param.default) {
                    switchEl.setAttribute('checked', '');
                }
                controlDiv.appendChild(switchEl);
                break;
            }

            default:
                return null;
        }

        return controlDiv;
    }
    generateParameterControls(benchmarkId) {
        // Deprecated: kept for backward compatibility.
        return '';
    }
    createParameterControl(benchmarkId, param) {
        const paramId = `${benchmarkId}-${param.id}`;
        const unit = param.unit ? ` ${param.unit}` : '';

        switch (param.type) {
            case 'slider':
                return `
                    <div class="parameter-control">
                        <sl-range
                            id="${paramId}"
                            label="${param.label}"
                            min="${param.min}"
                            max="${param.max}"
                            value="${param.default}"
                            step="${param.step}"
                            help-text="${param.description}">
                            <span slot="suffix">${param.default}${unit}</span>
                        </sl-range>
                    </div>
                `;

            case 'select':
                return `
                    <div class="parameter-control">
                        <sl-select id="${paramId}" label="${param.label}" value="${param.default}" help-text="${param.description}">
                            ${param.options.map(opt =>
                                `<sl-option value="${opt.value}">${opt.label}</sl-option>`
                            ).join('')}
                        </sl-select>
                    </div>
                `;

            case 'switch':
                return `
                    <div class="parameter-control">
                        <sl-switch id="${paramId}" ${param.default ? 'checked' : ''} help-text="${param.description}">
                            ${param.label}
                        </sl-switch>
                    </div>
                `;

            default:
                return '';
        }
    }
    setupParameterListeners(benchmarkId) {
        const params = this.getBenchmarkParameters(benchmarkId);

        params.forEach(param => {
            const element = document.getElementById(`${benchmarkId}-${param.id}`);
            if (!element) return;

            if (param.type === 'slider') {
                element.addEventListener('sl-input', (event) => {
                    const suffixElement = element.querySelector('[slot="suffix"]');
                    const unit = param.unit ? ` ${param.unit}` : '';
                    if (suffixElement) {
                        suffixElement.textContent = `${event.target.value}${unit}`;
                    }
                    const numericValue = parseFloat(event.target.value);
                    this.updateParameterValue(
                        benchmarkId,
                        param.id,
                        Number.isNaN(numericValue) ? param.default : numericValue
                    );
                });
            } else if (param.type === 'select') {
                element.addEventListener('sl-change', (event) => {
                    this.updateParameterValue(benchmarkId, param.id, event.target.value);
                });
            } else if (param.type === 'switch') {
                element.addEventListener('sl-change', (event) => {
                    this.updateParameterValue(benchmarkId, param.id, event.target.checked);
                });
            }
        });

        const resetButton = document.getElementById(`reset-params-${benchmarkId}`);
        if (resetButton) {
            resetButton.addEventListener('click', () => this.resetParametersToDefaults(benchmarkId));
        }

        const loadButton = document.getElementById(`load-preset-${benchmarkId}`);
        if (loadButton) {
            loadButton.addEventListener('click', () => this.showPresetLoader(benchmarkId));
        }

        const saveButton = document.getElementById(`save-preset-${benchmarkId}`);
        if (saveButton) {
            saveButton.addEventListener('click', () => this.showPresetSaver(benchmarkId));
        }
    }
    updateParameterValue(benchmarkId, paramId, value) {
        if (!this.benchmarkParameters.has(benchmarkId)) {
            this.benchmarkParameters.set(benchmarkId, new Map());
        }

        this.benchmarkParameters.get(benchmarkId).set(paramId, value);

        this.saveParametersToStorage(benchmarkId);

        console.log(`Parameter updated: ${benchmarkId}.${paramId} = ${value}`);
    }
    hasStoredParameters(benchmarkId) {
        try {
            return localStorage.getItem(`webgpu-benchmark-params-${benchmarkId}`) !== null;
        } catch (error) {
            console.warn('Unable to access localStorage for parameters:', error);
            return false;
        }
    }
    resetParametersToDefaults(benchmarkId) {
        const params = this.getBenchmarkParameters(benchmarkId);

        params.forEach(param => {
            const element = document.getElementById(`${benchmarkId}-${param.id}`);
            if (!element) return;

            if (param.type === 'slider') {
                element.value = param.default;
                const suffixElement = element.querySelector('[slot="suffix"]');
                const unit = param.unit ? ` ${param.unit}` : '';
                if (suffixElement) {
                    suffixElement.textContent = `${param.default}${unit}`;
                }
            } else if (param.type === 'select') {
                element.value = param.default;
            } else if (param.type === 'switch') {
                element.checked = param.default;
            }

            this.updateParameterValue(benchmarkId, param.id, param.default);
        });
    }
    saveParametersToStorage(benchmarkId) {
        const params = this.benchmarkParameters.get(benchmarkId);
        if (params) {
            const paramObj = Object.fromEntries(params);
            try {
                localStorage.setItem(`webgpu-benchmark-params-${benchmarkId}`, JSON.stringify(paramObj));
            } catch (error) {
                console.warn('Unable to persist benchmark parameters:', error);
            }
        }
    }
    loadParametersFromStorage(benchmarkId) {
        const stored = localStorage.getItem(`webgpu-benchmark-params-${benchmarkId}`);
        if (stored) {
            try {
                const paramObj = JSON.parse(stored);
                const paramMap = new Map(Object.entries(paramObj));
                this.benchmarkParameters.set(benchmarkId, paramMap);

                this.applyParametersToUI(benchmarkId, paramMap);
            } catch (error) {
                console.warn(`Failed to load parameters for ${benchmarkId}:`, error);
            }
        }
    }
    applyParametersToUI(benchmarkId, paramMap) {
        const params = this.getBenchmarkParameters(benchmarkId);

        params.forEach(param => {
            const element = document.getElementById(`${benchmarkId}-${param.id}`);
            const value = paramMap.get(param.id);

            if (!element || value === undefined) return;

            if (param.type === 'slider') {
                element.value = value;
                const suffixElement = element.querySelector('[slot="suffix"]');
                const unit = param.unit ? ` ${param.unit}` : '';
                if (suffixElement) {
                    suffixElement.textContent = `${value}${unit}`;
                }
            } else if (param.type === 'select') {
                element.value = value;
            } else if (param.type === 'switch') {
                element.checked = value;
            }
        });
    }
    showPresetLoader(benchmarkId) {
        alert('Preset loading feature coming soon!');
    }
    showPresetSaver(benchmarkId) {
        alert('Preset saving feature coming soon!');
    }
    setupEventListeners() {
        console.log('Event listeners will be added dynamically for selected benchmarks');
    }
    getConfiguration(benchmarkId) {
        this.quietMode = Boolean(document.getElementById('quiet-mode')?.checked);

        return {
            bufferSize: parseInt(document.getElementById('buffer-size').value) || 512,
            trackCount: parseInt(document.getElementById('track-count').value) || 128,
            iterations: parseInt(document.getElementById('iterations').value) || 100,
            warmup: parseInt(document.getElementById('warmup').value) || 3,
            parameters: this.getBenchmarkParameterValues(benchmarkId),
            quiet: this.quietMode
        };
    }
    async _executeBenchmark(benchmarkId, config) {
        const benchmarkCreator = this.benchmarkFactory[benchmarkId];
        if (!benchmarkCreator) {
            throw new Error(`Unknown benchmark type: ${benchmarkId}`);
        }

        const benchmark = benchmarkCreator(this.device, config);
        if (typeof benchmark.setQuiet === 'function') {
            benchmark.setQuiet(config.quiet);
        }
        const results = await benchmark.runBenchmark(config.iterations, config.warmup);

        return { benchmark, results };
    }
    async runBenchmark(benchmarkType) {
        if (!this.isInitialized) {
            console.error('WebGPU not initialized');
            return;
        }

        const config = this.getConfiguration(benchmarkType);
        const runButton = document.getElementById(`run-${benchmarkType}`);
        const exportButton = document.getElementById(`export-${benchmarkType}`);

        try {
            runButton.loading = true;
            runButton.disabled = true;
            exportButton.disabled = true;
            this.hideResults(benchmarkType);

            this.log(`Starting ${benchmarkType} benchmark with config:`, config);

            if (this.currentBenchmark && typeof this.currentBenchmark.cleanup === 'function') {
                try {
                    this.currentBenchmark.cleanup();
                } catch (cleanupError) {
                    console.warn('Error cleaning up previous benchmark:', cleanupError);
                }
            }

            const { benchmark, results } = await this._executeBenchmark(benchmarkType, config);

            this.currentBenchmark = benchmark;

            this.displayResults(benchmarkType, results);

            exportButton.disabled = false;

            this.log(`${benchmarkType} benchmark completed successfully`);

        } catch (error) {
            console.error(`${benchmarkType} benchmark failed:`, error);
            this.showError(benchmarkType, error.message);
        } finally {
            runButton.loading = false;
            runButton.disabled = false;
        }
    }
    displayResults(benchmarkType, results) {
        const resultsSection = document.getElementById(`${benchmarkType}-results`);

        resultsSection.style.display = 'block';

        const stats = results.statistics;
        document.getElementById(`${benchmarkType}-median`).textContent =
            Statistics.formatTime(stats.median);
        document.getElementById(`${benchmarkType}-p95`).textContent =
            Statistics.formatTime(stats.p95);
        document.getElementById(`${benchmarkType}-max`).textContent =
            Statistics.formatTime(stats.max);
        document.getElementById(`${benchmarkType}-min`).textContent =
            Statistics.formatTime(stats.min);

        this.updateValidationAndMetadata(benchmarkType, results);

        this.updateChart(benchmarkType, results);
    }
    updateValidationAndMetadata(benchmarkType, results) {
        if (!results.validation) return;

        const validationEl = document.getElementById(`${benchmarkType}-validation`);
        if (validationEl) {
            validationEl.textContent = results.validation.passed ? 'PASSED' : 'FAILED';
            validationEl.className = `stat-value ${results.validation.passed ? 'success' : 'error'}`;
        }

        const maxErrorEl = document.getElementById(`${benchmarkType}-max-error`);
        if (maxErrorEl && results.validation.maxError !== undefined) {
            maxErrorEl.textContent = results.validation.maxError.toExponential(3);
        }

        if (results.metadata) {
            const datasizeEl = document.getElementById(`${benchmarkType}-datasize`);
            if (datasizeEl && results.metadata.total_mb !== undefined) {
                datasizeEl.textContent = `${results.metadata.total_mb.toFixed(2)} MB`;
            }

            const memsizeEl = document.getElementById(`${benchmarkType}-memsize`);
            if (memsizeEl && results.metadata.sample_memory_mb !== undefined) {
                memsizeEl.textContent = `${results.metadata.sample_memory_mb.toFixed(1)} MB`;
            }

            const bandwidthEl = document.getElementById(`${benchmarkType}-bandwidth`);
            if (bandwidthEl && benchmarkType === 'randommemory' && results.metadata.total_samples) {
                const bytesRead = results.metadata.total_samples * CONSTANTS.BYTES_PER_FLOAT;
                const medianTimeMs = results.statistics.median;
                const medianTimeS = medianTimeMs / CONSTANTS.MS_TO_SECONDS;
                const bandwidth = (bytesRead / medianTimeS) / CONSTANTS.BYTES_TO_GB;
                bandwidthEl.textContent = `${bandwidth.toFixed(2)} GB/s`;
            }
        }
    }
    updateChart(benchmarkType, results) {
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not available, skipping chart creation');
            return;
        }

        const canvas = document.getElementById(`${benchmarkType}-chart`);
        const ctx = canvas.getContext('2d');

        if (this.charts.has(benchmarkType)) {
            this.charts.get(benchmarkType).destroy();
        }

        const histogram = Statistics.histogram(results.latencies, CONSTANTS.DEFAULT_CHART_BINS);
        if (!histogram.counts.length) {
            console.warn('No latency data available for histogram.');
            return;
        }

        const maxCount = Math.max(...histogram.counts);
        const avgCount = histogram.counts.reduce((sum, count) => sum + count, 0) / histogram.counts.length;
        const sortedCounts = [...histogram.counts].sort((a, b) => b - a);
        const secondHighest = sortedCounts[1] || 0;

        let yAxisMax = Math.min(
            maxCount * 0.2,
            avgCount * 3,
            secondHighest * 2
        );
        if (histogram.counts.length === 1 || yAxisMax <= 0) {
            yAxisMax = maxCount;
        }

        const chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: histogram.bins.slice(0, -1).map(bin =>
                    Statistics.formatTime(bin)
                ),
                datasets: [{
                    label: 'Frequency',
                    data: histogram.counts,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: `${benchmarkType.toUpperCase()} Benchmark - Latency Distribution`
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            afterBody: function(context) {
                                const total = histogram.counts.reduce((sum, count) => sum + count, 0);
                                const percentage = ((context[0].raw / total) * 100).toFixed(1);
                                return `${percentage}% of samples`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Latency'
                        },
                        ticks: {
                            maxRotation: 45,
                            callback: function(value, index) {
                                return index % 2 === 0 ? this.getLabelForValue(value) : '';
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        },
                        beginAtZero: true,
                        max: yAxisMax,
                        ticks: {
                            callback: function(value) {
                                return Number.isInteger(value) ? value : '';
                            }
                        }
                    }
                }
            }
        });

        this.charts.set(benchmarkType, chart);
    }
    hideResults(benchmarkType) {
        const resultsSection = document.getElementById(`${benchmarkType}-results`);
        resultsSection.style.display = 'none';
    }
    showError(benchmarkType, message) {
        console.error(`${benchmarkType} error:`, message);
        alert(`${benchmarkType} benchmark failed: ${message}`);
    }
    exportResults(benchmarkType) {
        if (!this.currentBenchmark || !this.currentBenchmark.getResults()) {
            console.error('No results to export');
            return;
        }

        try {
            const results = this.currentBenchmark.getResults();
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `webgpu_${benchmarkType}_${timestamp}.json`;

            this.currentBenchmark.exportResults(filename);
        } catch (error) {
            console.error('Failed to export results:', error);
            this.showError(benchmarkType, `Export failed: ${error.message}`);
        }
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded - charts will be disabled');
    }

    const app = new WebGPUBenchmarkApp();

    const initialized = await app.initialize();

    if (initialized) {
        console.log('WebGPU Audio Benchmark Suite ready!');
    } else {
        console.error('Failed to initialize WebGPU Audio Benchmark Suite');
    }
});
