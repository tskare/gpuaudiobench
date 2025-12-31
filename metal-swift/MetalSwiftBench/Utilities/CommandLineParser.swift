import Foundation

struct CommandLineParser<Configuration> {
    struct Option {
        enum Kind {
            case flag((inout Configuration) -> Void)
            case value(valueName: String, (String, inout Configuration) throws -> Void)
        }

        let names: [String]
        let help: String
        let allowsMultiple: Bool
        let kind: Kind

        init(names: [String], help: String, allowsMultiple: Bool = false, kind: Kind) {
            precondition(!names.isEmpty, "Option must declare at least one name")
            self.names = names
            self.help = help
            self.allowsMultiple = allowsMultiple
            self.kind = kind
        }
    }

    enum ParseError: LocalizedError {
        case unknownOption(String)
        case missingValue(option: String, valueName: String)
        case unexpectedValue(option: String)
        case validation(message: String)

        var errorDescription: String? {
            switch self {
            case .unknownOption(let token):
                return "Unknown option '\(token)'."
            case .missingValue(let option, let valueName):
                return "Missing value for \(option) (\(valueName))."
            case .unexpectedValue(let option):
                return "\(option) does not take a value."
            case .validation(let message):
                return message
            }
        }
    }

    private let options: [Option]
    private let positionalHandler: ((String, inout Configuration) throws -> Void)?

    init(options: [Option], positionalHandler: ((String, inout Configuration) throws -> Void)? = nil) {
        self.options = options
        self.positionalHandler = positionalHandler
    }

    func parse(into configuration: inout Configuration, arguments: [String]) throws {
        var seenOptions = Set<String>()
        var index = 0
        while index < arguments.count {
            let token = arguments[index]

            if token == "--" {
                index += 1
                while index < arguments.count {
                    try positionalHandler?(arguments[index], &configuration)
                    index += 1
                }
                break
            }

            if token.hasPrefix("-") {
                let (name, attachedValue) = splitOptionToken(token)
                guard let option = option(named: name) else {
                    throw ParseError.unknownOption(name)
                }

                if !option.allowsMultiple, let canonicalName = option.names.first {
                    if seenOptions.contains(canonicalName) {
                        throw ParseError.validation(message: "Option \(canonicalName) supplied multiple times.")
                    }
                    seenOptions.insert(canonicalName)
                }

                switch option.kind {
                case .flag(let action):
                    if let value = attachedValue {
                        throw ParseError.unexpectedValue(option: name + "=" + value)
                    }
                    action(&configuration)

                case .value(let valueName, let action):
                    let value: String
                    if let attached = attachedValue {
                        value = attached
                    } else {
                        let nextIndex = index + 1
                        guard nextIndex < arguments.count else {
                            throw ParseError.missingValue(option: name, valueName: valueName)
                        }
                        value = arguments[nextIndex]
                        index = nextIndex
                    }
                    try action(value, &configuration)
                }

                index += 1
                continue
            }

            if let positional = positionalHandler {
                try positional(token, &configuration)
            }
            index += 1
        }
    }

    private func option(named name: String) -> Option? {
        return options.first { $0.names.contains(name) }
    }

    private func splitOptionToken(_ token: String) -> (String, String?) {
        if let equalsIndex = token.firstIndex(of: "=") {
            let name = String(token[..<equalsIndex])
            let value = String(token[token.index(after: equalsIndex)...])
            return (name, value)
        }
        return (token, nil)
    }

    func helpText(executableName: String) -> String {
        let columnWidth = options
            .flatMap { $0.names }
            .map { $0.count }
            .max() ?? 0

        var helpLines: [String] = []
        helpLines.append("GPU Audio Benchmark Suite - Swift Implementation")
        helpLines.append("")
        helpLines.append("Usage: \(executableName) [options]")
        helpLines.append("")
        helpLines.append("Options:")

        for option in options {
            let names = option.names.joined(separator: ", ")
            let paddedNames = names.padding(toLength: columnWidth, withPad: " ", startingAt: 0)
            let description: String
            switch option.kind {
            case .flag:
                description = option.help
            case .value(let valueName, _):
                description = "\(option.help) (\(valueName))"
            }
            helpLines.append("  \(paddedNames)  \(description)")
        }

        return helpLines.joined(separator: "\n")
    }
}
