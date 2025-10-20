// Helpers for building benchmark configuration controls.
export class ParameterBuilder {
    static slider(id, label, min, max, defaultValue, step = 1, unit = '', description = '') {
        return {
            id,
            type: 'slider',
            label,
            min,
            max,
            default: defaultValue,
            step,
            unit,
            description
        };
    }

    static select(id, label, defaultValue, options, description = '') {
        return {
            id,
            type: 'select',
            label,
            default: defaultValue,
            options,
            description
        };
    }

    static switch(id, label, defaultValue, description = '') {
        return {
            id,
            type: 'switch',
            label,
            default: defaultValue,
            description
        };
    }

    static filterParams(cutoffDefault = 1000, qDefault = 0.707) {
        return [
            this.select('filterType', 'Filter Type', 'butterworth', [
                { value: 'butterworth', label: 'Butterworth' },
                { value: 'chebyshev1', label: 'Chebyshev Type I' },
                { value: 'chebyshev2', label: 'Chebyshev Type II' },
                { value: 'elliptic', label: 'Elliptic' }
            ], 'Type of IIR filter design'),

            this.slider('cutoffFreq', 'Cutoff Frequency', 100, 20000, cutoffDefault, 50, 'Hz', 'Filter cutoff frequency in Hz'),

            this.slider('qFactor', 'Q Factor', 0.1, 10, qDefault, 0.1, '', 'Filter resonance (quality factor)'),

            this.select('filterOrder', 'Filter Order', 2, [
                { value: 1, label: '1st Order' },
                { value: 2, label: '2nd Order (Biquad)' },
                { value: 4, label: '4th Order' },
                { value: 8, label: '8th Order' }
            ], 'Number of filter poles')
        ];
    }

    static convolutionParams(irLengthDefault = 512) {
        return [
            this.slider('irLength', 'Impulse Response Length', 64, 2048, irLengthDefault, 64, 'samples', 'Length of the convolution impulse response'),

            this.switch('useConstantMemory', 'Use Constant Memory', true, 'Use constant memory for impulse responses vs device memory'),

            this.select('windowFunction', 'Window Function', 'hann', [
                { value: 'hann', label: 'Hann Window' },
                { value: 'hamming', label: 'Hamming Window' },
                { value: 'blackman', label: 'Blackman Window' },
                { value: 'kaiser', label: 'Kaiser Window' }
            ], 'Window function for impulse response generation')
        ];
    }

    static fftParams(defaultSize = 1024) {
        const sizeOptions = [256, 512, 1024, 2048, 4096].map(value => ({ value, label: `${value}` }));

        if (!sizeOptions.some(option => option.value === defaultSize)) {
            sizeOptions.push({ value: defaultSize, label: `${defaultSize}` });
        }

        return [
            this.select('fftSize', 'FFT Size', defaultSize, sizeOptions, 'Number of samples processed per transform (power-of-two recommended)')
        ];
    }

    static memoryParams(memorySizeDefault = 128, minLoopDefault = 1000, maxLoopDefault = 48000) {
        return [
            this.slider('sampleMemorySize', 'Sample Memory Size', 32, 512, memorySizeDefault, 16, 'MB', 'Size of sample memory pool for granular synthesis'),

            this.slider('minLoopLength', 'Min Loop Length', 100, 5000, minLoopDefault, 100, 'samples', 'Minimum grain loop length'),

            this.slider('maxLoopLength', 'Max Loop Length', 1000, 96000, maxLoopDefault, 1000, 'samples', 'Maximum grain loop length'),

            this.slider('grainDensity', 'Grain Density', 0.1, 10.0, 1.0, 0.1, 'grains/ms', 'Density of granular synthesis grains')
        ];
    }

    static waveguideParams(lengthDefault = 100, accelDefault = true) {
        const baseParams = [
            this.slider('waveguideLength', 'Waveguide Length', 10, 1000, lengthDefault, 10, 'samples', 'Length of the digital waveguide delay line'),

            this.slider('inputTapPos', 'Input Tap Position', 0, 1, 0.1, 0.01, '', 'Input excitation position (0=bridge, 1=nut)'),

            this.slider('outputTapPos', 'Output Tap Position', 0, 1, 0.9, 0.01, '', 'Output pickup position (0=bridge, 1=nut)'),

            this.slider('reflectionCoeff', 'Reflection Coefficient', -1, 1, -0.99, 0.01, '', 'Boundary reflection coefficient'),

            this.slider('dampingFactor', 'Damping Factor', 0, 1, 0.9999, 0.0001, '', 'Damping coefficient (1.0 = no damping)')
        ];

        if (accelDefault !== null) {
            baseParams.push(
                this.switch('accelerationMode', 'Platform Acceleration', accelDefault, 'Enable platform-specific optimizations')
            );
        }

        return baseParams;
    }

    static fdtdParams(roomSizeDefault = 50) {
        return [
            this.slider('roomSizeX', 'Room Size X', 10, 100, roomSizeDefault, 5, 'grid points', 'Room dimension in X direction'),

            this.slider('roomSizeY', 'Room Size Y', 10, 100, roomSizeDefault, 5, 'grid points', 'Room dimension in Y direction'),

            this.slider('roomSizeZ', 'Room Size Z', 10, 100, roomSizeDefault, 5, 'grid points', 'Room dimension in Z direction'),

            this.slider('absorptionCoeff', 'Wall Absorption', 0, 1, 0.2, 0.01, '', 'Wall absorption coefficient (0=reflective, 1=absorptive)'),

            this.slider('soundSpeed', 'Sound Speed', 300, 400, 343, 1, 'm/s', 'Speed of sound in air'),

            this.slider('spatialStep', 'Spatial Step Size', 0.001, 0.1, 0.01, 0.001, 'm', 'Grid spacing for finite difference calculation')
        ];
    }

    static modalParams(numModesDefault = 1024, outputTracksDefault = 32) {
        return [
            this.slider('numModes', 'Number of Modes', 128, 8192, numModesDefault, 128, '', 'Number of resonant modes in the filter bank'),

            this.slider('outputTracks', 'Output Tracks', 1, 64, outputTracksDefault, 1, '', 'Number of output audio tracks'),

            this.slider('frequencySpread', 'Frequency Spread', 0.1, 5.0, 1.0, 0.1, '', 'Frequency distribution spread factor'),

            this.slider('dampingCoeff', 'Damping Coefficient', 0.001, 0.1, 0.01, 0.001, '', 'Modal damping coefficient')
        ];
    }
}
