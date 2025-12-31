import Foundation
import Metal
import MetalKit

struct CommandLineArgs {
    var bufferSize: Int = 512
    var benchmarkFilters: [String] = []
    var sampleRate: Int = 48000
    var trackCount: Int = 128
    var runCount: Int = 100
    var dawSimulation: Bool = false
    var dawSimulationMode: DAWSimulationMode = .spin
    var dawSimulationJitterUs: Double = 0
    var outputFile: String?
    var showHelp: Bool = false
    var listOnly: Bool = false
    var jsonOutput: Bool = false
    var verificationMode: ValidationMode = .full
    var captureEnabled: Bool = false
}

private func makeArgumentParser() -> CommandLineParser<CommandLineArgs> {
    typealias Parser = CommandLineParser<CommandLineArgs>
    typealias ParseError = Parser.ParseError

    func parsePositiveInt(_ value: String, option: String, minimum: Int = 1) throws -> Int {
        guard let parsed = Int(value), parsed >= minimum else {
            throw ParseError.validation(message: "Option \(option) expects an integer ≥ \(minimum). Received '\(value)'.")
        }
        return parsed
    }

    func parseNonNegativeDouble(_ value: String, option: String) throws -> Double {
        guard let parsed = Double(value), parsed >= 0 else {
            throw ParseError.validation(message: "Option \(option) expects a number ≥ 0. Received '\(value)'.")
        }
        return parsed
    }

    func appendFilters(_ value: String, into configuration: inout CommandLineArgs) {
        let filters = value
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        configuration.benchmarkFilters.append(contentsOf: filters)
    }

    let options: [Parser.Option] = [
        .init(
            names: ["--help", "-h"],
            help: "Show this help message",
            kind: .flag { config in config.showHelp = true }
        ),
        .init(
            names: ["--buffersize"],
            help: "Audio buffer size in samples (default: 512)",
            kind: .value(valueName: "size") { value, config in
                config.bufferSize = try parsePositiveInt(value, option: "--buffersize")
            }
        ),
        .init(
            names: ["--benchmarkFilter"],
            help: "Comma-separated benchmark filter list (substring, exact, or /regex/)",
            allowsMultiple: true,
            kind: .value(valueName: "filters") { value, config in
                appendFilters(value, into: &config)
            }
        ),
        .init(
            names: ["--benchmark"],
            help: "Deprecated alias for --benchmarkFilter",
            allowsMultiple: true,
            kind: .value(valueName: "filters") { value, config in
                print("Warning: --benchmark is deprecated. Use --benchmarkFilter instead.")
                appendFilters(value, into: &config)
            }
        ),
        .init(
            names: ["--list"],
            help: "List available benchmarks and exit",
            kind: .flag { config in config.listOnly = true }
        ),
        .init(
            names: ["--json"],
            help: "Emit JSON summary to stdout (and file if --outputfile provided)",
            kind: .flag { config in config.jsonOutput = true }
        ),
        .init(
            names: ["--fs"],
            help: "Sample rate in Hz (default: 48000)",
            kind: .value(valueName: "rate") { value, config in
                config.sampleRate = try parsePositiveInt(value, option: "--fs")
            }
        ),
        .init(
            names: ["--ntracks"],
            help: "Number of audio tracks (default: 128)",
            kind: .value(valueName: "count") { value, config in
                config.trackCount = try parsePositiveInt(value, option: "--ntracks")
            }
        ),
        .init(
            names: ["--nruns"],
            help: "Number of benchmark runs (default: 100)",
            kind: .value(valueName: "count") { value, config in
                config.runCount = try parsePositiveInt(value, option: "--nruns")
            }
        ),
        .init(
            names: ["--dawsim"],
            help: "Simulate DAW-style scheduling and GPU interop",
            kind: .flag { config in config.dawSimulation = true }
        ),
        .init(
            names: ["--dawsim-mode"],
            help: "DAW simulation mode: spin | sleep (default: spin)",
            kind: .value(valueName: "mode") { value, config in
                switch value.lowercased() {
                case "spin":
                    config.dawSimulationMode = .spin
                case "sleep":
                    config.dawSimulationMode = .sleep
                default:
                    throw ParseError.validation(message: "Unknown DAW simulation mode '\(value)'. Expected spin | sleep.")
                }
            }
        ),
        .init(
            names: ["--dawsim-jitter-us"],
            help: "DAW simulation jitter in microseconds (default: 0)",
            kind: .value(valueName: "us") { value, config in
                config.dawSimulationJitterUs = try parseNonNegativeDouble(value, option: "--dawsim-jitter-us")
            }
        ),
        .init(
            names: ["--outputfile"],
            help: "Path for CSV/JSON output",
            kind: .value(valueName: "path") { value, config in
                config.outputFile = value
            }
        ),
        .init(
            names: ["--verification"],
            help: "Validation mode: none | spot | full (default: full)",
            kind: .value(valueName: "mode") { value, config in
                switch value.lowercased() {
                case "none":
                    config.verificationMode = .none
                case "spot":
                    config.verificationMode = .spot
                case "full":
                    config.verificationMode = .full
                default:
                    throw ParseError.validation(message: "Unknown verification mode '\(value)'. Expected none | spot | full.")
                }
            }
        ),
        .init(
            names: ["--capture"],
            help: "Capture GPU trace for the benchmark run",
            kind: .flag { config in config.captureEnabled = true }
        )
    ]

    return CommandLineParser(options: options)
}

private func printAvailableBenchmarks(includeLeadingBlankLine: Bool = false) {
    if includeLeadingBlankLine {
        print("")
    }
    print("Available benchmarks:")
    for (name, description) in BenchmarkRegistry.listWithDescriptions() {
        print("  \(name) - \(description)")
    }
}

func main() {
    let parser = makeArgumentParser()
    var args = CommandLineArgs()
    let rawArguments = Array(CommandLine.arguments.dropFirst())

    do {
        try parser.parse(into: &args, arguments: rawArguments)
    } catch {
        let executable = CommandLine.arguments.first ?? "MetalSwiftBench"
        print("Error: \(error.localizedDescription)")
        print("")
        print(parser.helpText(executableName: executable))
        exit(1)
    }

    if args.showHelp {
        let executable = CommandLine.arguments.first ?? "MetalSwiftBench"
        print(parser.helpText(executableName: executable))
        printAvailableBenchmarks(includeLeadingBlankLine: true)
        exit(0)
    }
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Error: Metal is not supported on this device")
        exit(1)
    }
    
    print("Metal device: \(device.name)")
    print("Buffer size: \(args.bufferSize), Sample rate: \(args.sampleRate), Tracks: \(args.trackCount)")

    if args.listOnly {
        printAvailableBenchmarks(includeLeadingBlankLine: true)
        exit(0)
    }
    
    let allBenchmarks = BenchmarkType.allCases
    let filters = args.benchmarkFilters
    let selected: [BenchmarkType]
    if filters.isEmpty {
        selected = allBenchmarks
        print("Selected benchmarks: ALL (\(selected.count))")
    } else {
        var matched = Set<BenchmarkType>()
        for bench in allBenchmarks {
            let name = bench.rawValue
            for token in filters {
                if token.hasPrefix("/") && token.hasSuffix("/") && token.count > 2 {
                    let pattern = String(token.dropFirst().dropLast())
                    do {
                        let regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
                        let range = NSRange(location: 0, length: name.utf16.count)
                        if regex.firstMatch(in: name, options: [], range: range) != nil {
                            matched.insert(bench)
                        }
                    } catch {
                        print("Invalid regex pattern: \(token) -> \(error)")
                    }
                } else if name.caseInsensitiveCompare(token) == .orderedSame {
                    matched.insert(bench)
                } else if name.lowercased().contains(token.lowercased()) {
                    matched.insert(bench)
                }
            }
        }
        selected = Array(allBenchmarks.filter { matched.contains($0) })
        print("Filters: \(filters.joined(separator: ",")) -> Matched: \(selected.map { $0.rawValue }.joined(separator: ", "))")
    }

    if selected.isEmpty {
        print("No benchmarks matched the provided filters: \(filters)")
        printAvailableBenchmarks()
        exit(1)
    }

    var outputLines: [String] = []
    var jsonArray: [[String: Any]] = []
    let wantCSV = !args.jsonOutput
    if wantCSV, let _ = args.outputFile {
        outputLines.append("benchmark,median_ms,p95_ms,p99_ms,max_ms,mean_ms,stddev_ms,count")
    }

    for (idx, benchmarkType) in selected.enumerated() {
        print("\n=== [\(idx + 1)/\(selected.count)] Running benchmark: \(benchmarkType.rawValue) ===")
        do {
            var benchmark = try benchmarkType.createBenchmark(
                device: device,
                bufferSize: args.bufferSize,
                trackCount: args.trackCount
            )
            benchmark.validationMode = args.verificationMode
            if args.dawSimulation {
                let jitterSeconds = args.dawSimulationJitterUs / 1_000_000
                benchmark.dawSimulator = DAWSimulator(
                    bufferDuration: Double(args.bufferSize) / Double(args.sampleRate),
                    mode: args.dawSimulationMode,
                    jitterSeconds: jitterSeconds
                )
            }

            var didStartCapture = false
            if args.captureEnabled {
                if #available(macOS 10.13, *) {
                    let captureManager = MTLCaptureManager.shared()
                    if captureManager.isCapturing {
                        captureManager.stopCapture()
                    }
                    let descriptor = MTLCaptureDescriptor()
                    descriptor.captureObject = benchmark.commandQueue
                    descriptor.destination = .developerTools
                    do {
                        try captureManager.startCapture(with: descriptor)
                        didStartCapture = true
                    } catch {
                        print("Warning: failed to start Metal capture: \(error)")
                    }
                } else {
                    print("Warning: Metal capture is not supported on this platform; ignoring --capture")
                }
            }

            defer {
                if didStartCapture {
                    if #available(macOS 10.13, *) {
                        MTLCaptureManager.shared().stopCapture()
                    }
                }
            }
            print("Setting up benchmark...")
            try benchmark.setup()
            print("Running \(args.runCount) iterations...")
            let result = try benchmark.runBenchmark(iterations: args.runCount, warmupIterations: 3)
            print("\n" + result.formatSummary())

            let deadlineTracker = DeadlineTracker(bufferSize: args.bufferSize, sampleRate: args.sampleRate)
            let missRate = deadlineTracker.deadlineMissRate(in: result)
            if missRate > 0 {
                print("Warning: \(String(format: "%.1f%%", missRate)) of runs missed the real-time deadline of \(String(format: "%.2f", deadlineTracker.deadlineMs))ms")
            }

            let stats = result.statistics
            if wantCSV {
                if let _ = args.outputFile {
                    outputLines.append("\(benchmarkType.rawValue),\(stats.median),\(stats.p95),\(stats.p99),\(stats.max),\(stats.mean),\(stats.standardDeviation),\(stats.count)")
                }
            } else {
                let entry: [String: Any] = [
                    "benchmark": benchmarkType.rawValue,
                    "median_ms": stats.median,
                    "p95_ms": stats.p95,
                    "p99_ms": stats.p99,
                    "max_ms": stats.max,
                    "mean_ms": stats.mean,
                    "stddev_ms": stats.standardDeviation,
                    "count": stats.count,
                    "metadata": result.metadata,
                    "device": device.name
                ]
                jsonArray.append(entry)
            }

            benchmark.cleanup()
        } catch {
            print("Error running benchmark \(benchmarkType.rawValue): \(error)")
        }
    }

    if args.jsonOutput {
        if let jsonData = try? JSONSerialization.data(withJSONObject: jsonArray, options: [.prettyPrinted]) {
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                print("\nJSON Results:\n" + jsonString)
            }
            if let outputFile = args.outputFile {
                do {
                    try jsonData.write(to: URL(fileURLWithPath: outputFile))
                    print("\nJSON written to: \(outputFile)")
                } catch {
                    print("Error writing JSON output file: \(error)")
                }
            }
        } else {
            print("Failed to serialize JSON results")
        }
    } else if let outputFile = args.outputFile {
        do {
            try (outputLines.joined(separator: "\n") + "\n").write(toFile: outputFile, atomically: true, encoding: .utf8)
            print("\nCSV written to: \(outputFile)")
        } catch {
            print("Error writing CSV output file: \(error)")
        }
    }
}

main()
