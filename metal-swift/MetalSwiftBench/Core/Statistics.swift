//
//  Statistics.swift
//  MetalSwiftBench
//
//  Statistical analysis utilities for benchmark results
//

import Foundation

struct Statistics {
    let values: [Double]
    
    init(values: [Double]) {
        self.values = values.sorted()
    }
    
    var count: Int {
        return values.count
    }
    
    var mean: Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }
    
    var median: Double {
        guard !values.isEmpty else { return 0 }
        let middle = values.count / 2
        if values.count % 2 == 0 {
            return (values[middle - 1] + values[middle]) / 2
        } else {
            return values[middle]
        }
    }
    
    var standardDeviation: Double {
        guard values.count > 1 else { return 0 }
        let avg = mean
        let variance = values.reduce(0) { $0 + pow($1 - avg, 2) } / Double(values.count - 1)
        return sqrt(variance)
    }
    
    func percentile(_ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        guard p >= 0 && p <= 100 else { return 0 }
        
        let index = Double(values.count - 1) * (p / 100)
        let lower = Int(floor(index))
        let upper = Int(ceil(index))
        let weight = index - Double(lower)
        
        if lower == upper {
            return values[lower]
        } else {
            return values[lower] * (1 - weight) + values[upper] * weight
        }
    }
    
    var p95: Double {
        return percentile(95)
    }
    
    var p99: Double {
        return percentile(99)
    }
    
    var min: Double {
        return values.first ?? 0
    }
    
    var max: Double {
        return values.last ?? 0
    }
}

extension BenchmarkResult {
    var statistics: Statistics {
        return Statistics(values: latencies.map { $0 * 1000 }) // Convert to milliseconds
    }

    var gpuStatistics: Statistics? {
        let nonZero = gpuLatencies.filter { $0 > 0 }
        guard !nonZero.isEmpty else { return nil }
        return Statistics(values: nonZero.map { $0 * 1000 })
    }
    
    func formatSummary() -> String {
        let stats = statistics
        var summary = String(format: """
            Latency Statistics (ms):
              Median: %.3f
              P95:    %.3f
              P99:    %.3f
              Max:    %.3f
              Mean:   %.3f
              StdDev: %.3f
              Count:  %d
            """,
            stats.median,
            stats.p95,
            stats.p99,
            stats.max,
            stats.mean,
            stats.standardDeviation,
            stats.count
        )

        if let gpuStats = gpuStatistics {
            summary += String(format: """
            GPU Execution (ms):
              Median: %.3f
              P95:    %.3f
              P99:    %.3f
              Max:    %.3f
              Mean:   %.3f
              StdDev: %.3f
              Samples:%4d
            """,
            gpuStats.median,
            gpuStats.p95,
            gpuStats.p99,
            gpuStats.max,
            gpuStats.mean,
            gpuStats.standardDeviation,
            gpuStats.count)
        }

        return summary
    }
    
    func toCSV() -> String {
        let stats = statistics
        return String(format: "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d",
            stats.median,
            stats.p95,
            stats.p99,
            stats.max,
            stats.mean,
            stats.standardDeviation,
            stats.count
        )
    }
    
    func toJSON() -> String? {
        let stats = statistics
        var dict: [String: Any] = [
            "median_ms": stats.median,
            "p95_ms": stats.p95,
            "p99_ms": stats.p99,
            "max_ms": stats.max,
            "mean_ms": stats.mean,
            "stddev_ms": stats.standardDeviation,
            "count": stats.count,
            "metadata": metadata
        ]

        if let gpuStats = gpuStatistics {
            dict["gpu_stats"] = [
                "median_ms": gpuStats.median,
                "p95_ms": gpuStats.p95,
                "p99_ms": gpuStats.p99,
                "max_ms": gpuStats.max,
                "mean_ms": gpuStats.mean,
                "stddev_ms": gpuStats.standardDeviation,
                "count": gpuStats.count
            ]
        }
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            return nil
        }
        
        return jsonString
    }
}

// Helper for tracking real-time deadlines
struct DeadlineTracker {
    let bufferSize: Int
    let sampleRate: Int
    
    var bufferDuration: TimeInterval {
        return Double(bufferSize) / Double(sampleRate)
    }
    
    var deadlineMs: Double {
        return bufferDuration * 1000
    }
    
    func missedDeadlines(in result: BenchmarkResult) -> Int {
        return result.latencies.filter { $0 > bufferDuration }.count
    }
    
    func deadlineMissRate(in result: BenchmarkResult) -> Double {
        guard !result.latencies.isEmpty else { return 0 }
        return Double(missedDeadlines(in: result)) / Double(result.latencies.count) * 100
    }
}
