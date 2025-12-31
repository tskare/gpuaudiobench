export class Statistics {
    static calculate(values) {
        if (!Array.isArray(values) || values.length === 0) {
            return {
                count: 0,
                min: 0,
                max: 0,
                mean: 0,
                median: 0,
                p95: 0,
                p99: 0,
                stddev: 0,
                variance: 0
            };
        }

        const sorted = [...values].sort((a, b) => a - b);
        const count = sorted.length;

        const min = sorted[0];
        const max = sorted[count - 1];
        const sum = values.reduce((acc, val) => acc + val, 0);
        const mean = sum / count;

        const median = count % 2 === 0
            ? (sorted[Math.floor(count / 2) - 1] + sorted[Math.floor(count / 2)]) / 2
            : sorted[Math.floor(count / 2)];

        const p95 = this.percentile(sorted, 95);
        const p99 = this.percentile(sorted, 99);

        const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / count;
        const stddev = Math.sqrt(variance);

        return {
            count,
            min,
            max,
            mean,
            median,
            p95,
            p99,
            stddev,
            variance
        };
    }
    static percentile(sortedArray, percentile) {
        if (sortedArray.length === 0) return 0;
        if (percentile <= 0) return sortedArray[0];
        if (percentile >= 100) return sortedArray[sortedArray.length - 1];

        const index = (percentile / 100) * (sortedArray.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);

        if (lower === upper) {
            return sortedArray[lower];
        }

        const weight = index - lower;
        return sortedArray[lower] * (1 - weight) + sortedArray[upper] * weight;
    }
    static formatTime(milliseconds) {
        if (milliseconds < 0.001) {
            return `${(milliseconds * 1000000).toFixed(2)} ns`;
        } else if (milliseconds < 1) {
            return `${(milliseconds * 1000).toFixed(2)} μs`;
        } else if (milliseconds < 1000) {
            return `${milliseconds.toFixed(3)} ms`;
        } else {
            return `${(milliseconds / 1000).toFixed(2)} s`;
        }
    }
    static histogram(values, bins = 20) {
        if (values.length === 0) return { bins: [], counts: [], binWidth: 0 };

        const min = Math.min(...values);
        const max = Math.max(...values);
        if (max === min) {
            // Degenerate case: all values identical
            const counts = new Array(bins).fill(0);
            counts[0] = values.length;
            return {
                bins: Array.from({ length: bins + 1 }, () => min),
                counts,
                binWidth: 0
            };
        }
        const binWidth = (max - min) / bins;

        const binEdges = Array.from({ length: bins + 1 }, (_, i) => min + i * binWidth);
        const counts = new Array(bins).fill(0);

        for (const value of values) {
            let binIndex = Math.floor((value - min) / binWidth);
            if (binIndex >= bins) binIndex = bins - 1; // Handle edge case
            counts[binIndex]++;
        }

        return {
            bins: binEdges,
            counts,
            binWidth
        };
    }
    static calculateErrorMetrics(gpuData, referenceData, tolerance) {
        if (!referenceData) {
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                tolerance,
                errorCount: 0,
                samplesChecked: 0,
                message: 'No reference data available'
            };
        }

        if (gpuData.length !== referenceData.length) {
            const samplesChecked = Math.min(gpuData.length, referenceData.length);
            return {
                passed: false,
                maxError: Infinity,
                meanError: Infinity,
                tolerance,
                errorCount: 0,
                samplesChecked,
                message: `Length mismatch: got ${gpuData.length}, expected ${referenceData.length}`
            };
        }

        let maxError = 0;
        let totalError = 0;
        let errorCount = 0;

        for (let i = 0; i < referenceData.length; i++) {
            const error = Math.abs(gpuData[i] - referenceData[i]);
            if (error > maxError) {
                maxError = error;
            }
            totalError += error;
            if (error > tolerance) {
                errorCount++;
            }
        }

        const meanError = referenceData.length > 0 ? totalError / referenceData.length : 0;
        const passed = maxError <= tolerance;

        return {
            passed,
            maxError,
            meanError,
            tolerance,
            errorCount,
            samplesChecked: referenceData.length,
            message: passed
                ? `Validation passed (max error ${maxError.toExponential(3)})`
                : `Max error ${maxError.toExponential(3)} exceeds tolerance ${tolerance}`
        };
    }
    static detectOutliers(values) {
        if (values.length < 4) return [];

        const sorted = [...values].sort((a, b) => a - b);
        const q1 = this.percentile(sorted, 25);
        const q3 = this.percentile(sorted, 75);
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;

        return values.filter(value => value < lowerBound || value > upperBound);
    }
    static coefficientOfVariation(values) {
        const stats = this.calculate(values);
        return stats.mean !== 0 ? stats.stddev / stats.mean : 0;
    }
    static compare(baseline, comparison) {
        const baseStats = this.calculate(baseline);
        const compStats = this.calculate(comparison);

        const medianImprovement = (baseStats.median - compStats.median) / baseStats.median;
        const meanImprovement = (baseStats.mean - compStats.mean) / baseStats.mean;

        return {
            baseline: baseStats,
            comparison: compStats,
            improvement: {
                median: medianImprovement,
                mean: meanImprovement,
                medianPercent: medianImprovement * 100,
                meanPercent: meanImprovement * 100
            }
        };
    }
    static toCSV(statisticsArray, includeHeaders = true) {
        if (statisticsArray.length === 0) return '';

        const headers = ['benchmark', 'count', 'min', 'max', 'mean', 'median', 'p95', 'p99', 'stddev'];
        const rows = [];

        if (includeHeaders) {
            rows.push(headers.join(','));
        }

        for (const stats of statisticsArray) {
            const row = [
                stats.benchmark || 'unknown',
                stats.count,
                stats.min.toFixed(6),
                stats.max.toFixed(6),
                stats.mean.toFixed(6),
                stats.median.toFixed(6),
                stats.p95.toFixed(6),
                stats.p99.toFixed(6),
                stats.stddev.toFixed(6)
            ];
            rows.push(row.join(','));
        }

        return rows.join('\n');
    }
    static summarize(results) {
        const stats = this.calculate(results.latencies);
        const outliers = this.detectOutliers(results.latencies);
        const cv = this.coefficientOfVariation(results.latencies);

        return {
            benchmark: results.benchmark,
            timestamp: new Date(results.metadata?.timestamp || Date.now()).toISOString(),
            configuration: {
                bufferSize: results.metadata?.bufferSize,
                trackCount: results.metadata?.trackCount,
                totalSamples: results.metadata?.totalSamples,
                iterations: results.metadata?.iterations
            },
            performance: {
                median: this.formatTime(stats.median),
                p95: this.formatTime(stats.p95),
                max: this.formatTime(stats.max),
                min: this.formatTime(stats.min),
                mean: this.formatTime(stats.mean),
                stddev: this.formatTime(stats.stddev)
            },
            quality: {
                outlierCount: outliers.length,
                outlierPercentage: (outliers.length / results.latencies.length) * 100,
                coefficientOfVariation: cv,
                stability: cv < 0.1 ? 'excellent' : cv < 0.2 ? 'good' : cv < 0.5 ? 'fair' : 'poor'
            },
            validation: results.validation || { passed: true, message: 'No validation available' }
        };
    }
}
