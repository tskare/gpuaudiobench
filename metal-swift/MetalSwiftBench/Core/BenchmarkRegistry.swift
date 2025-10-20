import Foundation
import Metal

enum BenchmarkType: String, CaseIterable {
    case noOp = "NoOp"
    case dataCopy0199 = "datacopy0199"
    case dataCopy2080 = "datacopy2080"
    case dataCopy5050 = "datacopy5050"
    case dataCopy8020 = "datacopy8020"
    case dataCopy9901 = "datacopy9901"
    case gain = "gain"
    case gainStats = "GainStats"
    case iirFilter = "IIRFilter"
    case modalFilterBank = "ModalFilterBank"
    case randomMemoryRead = "RndMemRead"
    case convolution1D = "Conv1D"
    case convolution1DAccel = "Conv1D_accel"
    case dwg1DNaive = "DWG1DNaive"
    case dwg1DAccel = "DWG1DAccel"
    case fdtd3D = "FDTD3D"
    case fft1D = "FFT1D"
    
    var displayName: String {
        switch self {
        case .noOp:
            return "No-Op Kernel Launch"
        case .dataCopy0199:
            return "Data Transfer (1% in, 99% out)"
        case .dataCopy2080:
            return "Data Transfer (20% in, 80% out)"
        case .dataCopy5050:
            return "Data Transfer (50% in, 50% out)"
        case .dataCopy8020:
            return "Data Transfer (80% in, 20% out)"
        case .dataCopy9901:
            return "Data Transfer (99% in, 1% out)"
        case .gain:
            return "Gain Processing"
        case .gainStats:
            return "Gain Processing with Statistics"
        case .iirFilter:
            return "IIR Filter"
        case .modalFilterBank:
            return "Modal Filter Bank"
        case .randomMemoryRead:
            return "Random Memory Read"
        case .convolution1D:
            return "1D Convolution"
        case .convolution1DAccel:
            return "1D Convolution (Accelerated)"
        case .dwg1DNaive:
            return "Digital Waveguide 1D (Naive)"
        case .dwg1DAccel:
            return "Digital Waveguide 1D (Accelerated)"
        case .fdtd3D:
            return "3D FDTD Room Acoustics"
        case .fft1D:
            return "1D FFT Processing"
        }
    }
    
    func createBenchmark(device: MTLDevice, bufferSize: Int, trackCount: Int) throws -> GPUABenchmark {
        switch self {
        case .noOp:
            return try NoOpBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .dataCopy0199:
            return try DataTransferBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount, inputRatio: 0.01, outputRatio: 0.99)
        case .dataCopy2080:
            return try DataTransferBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount, inputRatio: 0.20, outputRatio: 0.80)
        case .dataCopy5050:
            return try DataTransferBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount, inputRatio: 0.50, outputRatio: 0.50)
        case .dataCopy8020:
            return try DataTransferBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount, inputRatio: 0.80, outputRatio: 0.20)
        case .dataCopy9901:
            return try DataTransferBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount, inputRatio: 0.99, outputRatio: 0.01)
        case .gain:
            return try GainBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .gainStats:
            return try GainStatsBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .iirFilter:
            return try IIRFilterBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .modalFilterBank:
            return try ModalFilterBankBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .randomMemoryRead:
            return try RandomMemoryReadBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .convolution1D:
            return try Convolution1DBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .convolution1DAccel:
            return try Convolution1DAccelBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .dwg1DNaive:
            return try DWG1DNaiveBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .dwg1DAccel:
            return try DWG1DAccelBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .fdtd3D:
            return try FDTD3DBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        case .fft1D:
            return try FFTBenchmark(device: device, bufferSize: bufferSize, trackCount: trackCount)
        }
    }
}


struct BenchmarkRegistry {
    static func benchmark(named name: String) -> BenchmarkType? {
        return BenchmarkType.allCases.first { $0.rawValue.lowercased() == name.lowercased() }
    }

    static func listAvailable() -> [String] {
        return BenchmarkType.allCases.map { $0.rawValue }
    }

    static func listWithDescriptions() -> [(name: String, description: String)] {
        return BenchmarkType.allCases.map { ($0.rawValue, $0.displayName) }
    }
}
