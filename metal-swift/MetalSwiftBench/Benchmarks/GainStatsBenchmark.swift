//
//  GainStatsBenchmark.swift
//  MetalSwiftBench
//
//  Gain processing with statistics calculation benchmark
//  Applies gain while computing mean and max statistics per track
//

import Foundation
import Metal

final class GainStatsBenchmark: BaseBenchmark {
    private var inputBuffer: MTLBuffer?
    private var outputBuffer: MTLBuffer?
    private var statsBuffer: MTLBuffer?
    private let gainValue: Float = 2.0

    // CPU reference data
    private var hostInputBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenBuffer: UnsafeMutablePointer<Float>?
    private var cpuGoldenStats: UnsafeMutablePointer<Float>?

    required init(device: MTLDevice, bufferSize: Int, trackCount: Int) throws {
        try super.init(device: device, bufferSize: bufferSize, trackCount: trackCount)
        self.kernelName = "BenchmarkGainStats"
    }

    override func setup() throws {
        try super.setup()

        // Size buffers so each track writes contiguous stats pairs and audio frames.
        let audioBufferSizeBytes = trackCount * bufferSize * MemoryLayout<Float>.size
        let statsBufferSizeBytes = trackCount * 2 * MemoryLayout<Float>.size // [mean, max] per track

        // Create input buffer
        guard let inBuffer = device.makeBuffer(length: audioBufferSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: audioBufferSizeBytes)
        }
        inputBuffer = inBuffer

        // Create output buffer
        guard let outBuffer = device.makeBuffer(length: audioBufferSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: audioBufferSizeBytes)
        }
        outputBuffer = outBuffer

        // Create statistics buffer
        guard let statsBuf = device.makeBuffer(length: statsBufferSizeBytes, options: .storageModeShared) else {
            throw BenchmarkError.bufferCreationFailed(size: statsBufferSizeBytes)
        }
        statsBuffer = statsBuf

        // Mirror GPU layout on the host to cross-check gain and stats logic.
        hostInputBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * bufferSize)
        cpuGoldenBuffer = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * bufferSize)
        cpuGoldenStats = UnsafeMutablePointer<Float>.allocate(capacity: trackCount * 2)

        // Initialize input buffer with random audio data
        let inputPointer = inBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        for i in 0..<(trackCount * bufferSize) {
            let value = Float.random(in: 0...1)
            inputPointer[i] = value
            hostInputBuffer![i] = value
        }

        // Clear outputs so stale GPU samples cannot contaminate the stats.
        memset(outBuffer.contents(), 0, audioBufferSizeBytes)
        memset(statsBuf.contents(), 0, statsBufferSizeBytes)
        memset(cpuGoldenBuffer, 0, audioBufferSizeBytes)
        memset(cpuGoldenStats, 0, statsBufferSizeBytes)
        
        // Generate the CPU gain + statistics baseline for validation.
        calculateCPUGoldenReference()
    }

    private func calculateCPUGoldenReference() {
        // Process each track
        for track in 0..<trackCount {
            let startIdx = track * bufferSize
            var trackMean: Float = 0.0
            var trackMax: Float = -Float.greatestFiniteMagnitude

            // Process all samples in this track
            for i in 0..<bufferSize {
                let idx = startIdx + i
                let sample = hostInputBuffer![idx]

                // Apply gain
                cpuGoldenBuffer![idx] = gainValue * sample

                // Accumulate statistics on original sample
                trackMean += sample
                if sample > trackMax {
                    trackMax = sample
                }
            }

            // Finalize statistics
            trackMean /= Float(bufferSize)

            // Store statistics: [mean, max] per track
            cpuGoldenStats![track * 2 + 0] = trackMean
            cpuGoldenStats![track * 2 + 1] = trackMax
        }
    }

    override func performBenchmarkIteration() throws {
        guard let inputBuffer = inputBuffer,
              let outputBuffer = outputBuffer,
              let statsBuffer = statsBuffer else {
            throw BenchmarkError.pipelineCreationFailed
        }

        let params = makeParams(gainValue: gainValue)

        try executeKernelWithParams(
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            parameters: params,
            parameterIndex: 3,
            additionalBuffers: [(statsBuffer, 2)],
            label: "GainStats"
        )
    }

    override func runBenchmark(iterations: Int, warmupIterations: Int = 3) throws -> BenchmarkResult {
        let result = try super.runBenchmark(iterations: iterations, warmupIterations: warmupIterations)

        var enhancedMetadata = result.metadata
        enhancedMetadata["benchmark"] = "GainStats"
        enhancedMetadata["gainValue"] = gainValue
        enhancedMetadata["totalSamples"] = bufferSize * trackCount

        if shouldValidate() {
            let verificationResult = verifyAgainstGoldenReference()
            enhancedMetadata["verificationPassed"] = verificationResult.passed
            enhancedMetadata["maxAudioError"] = verificationResult.maxAudioError
            enhancedMetadata["maxStatsError"] = verificationResult.maxStatsError
            enhancedMetadata["meanAudioError"] = verificationResult.meanAudioError
            enhancedMetadata["meanStatsError"] = verificationResult.meanStatsError
            enhancedMetadata["verificationSampleCountAudio"] = verificationResult.audioSampleCount
            enhancedMetadata["verificationSampleCountStats"] = verificationResult.statsSampleCount
        } else {
            enhancedMetadata["verificationSkipped"] = true
        }

        return BenchmarkResult(latencies: result.latencies, gpuLatencies: result.gpuLatencies, metadata: enhancedMetadata)
    }

    private func verifyAgainstGoldenReference() -> (passed: Bool, maxAudioError: Float, maxStatsError: Float, meanAudioError: Float, meanStatsError: Float, audioSampleCount: Int, statsSampleCount: Int) {
        guard let outputBuffer = self.outputBuffer,
              let statsBuffer = self.statsBuffer else {
            return (false, Float.infinity, Float.infinity, Float.infinity, Float.infinity, 0, 0)
        }

        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * bufferSize)
        let statsPointer = statsBuffer.contents().bindMemory(to: Float.self, capacity: trackCount * 2)
        let tolerance: Float = 1e-5

        let audioResult = compareWithGolden(
            gpuPointer: outputPointer,
            cpuPointer: cpuGoldenBuffer!,
            sampleCount: trackCount * bufferSize,
            tolerance: tolerance
        )

        let statsResult = compareWithGolden(
            gpuPointer: statsPointer,
            cpuPointer: cpuGoldenStats!,
            sampleCount: trackCount * 2,
            tolerance: tolerance
        )

        let passed = audioResult.passed && statsResult.passed
        return (
            passed,
            audioResult.maxError,
            statsResult.maxError,
            audioResult.meanError,
            statsResult.meanError,
            audioResult.sampleCount,
            statsResult.sampleCount
        )
    }

    override func cleanup() {
        inputBuffer = nil
        outputBuffer = nil
        statsBuffer = nil
    }

    deinit {
        hostInputBuffer?.deallocate()
        cpuGoldenBuffer?.deallocate()
        cpuGoldenStats?.deallocate()
    }
}
