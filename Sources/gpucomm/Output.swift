import Foundation

enum OutputFormat: String {
    case human
    case json
    case jsonl
    case csv
}

func csvEscape(_ s: String) -> String {
    if s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r") {
        return "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
    return s
}

func printCSV(header: [String], rows: [[String]]) {
    print(header.map(csvEscape).joined(separator: ","))
    for row in rows {
        print(row.map(csvEscape).joined(separator: ","))
    }
}

struct OutputOptions {
    static func parse(_ reader: inout ArgReader, defaultFormat: OutputFormat, jsonImplies: OutputFormat) -> OutputFormat {
        let formatRaw = reader.popValue(for: "--format")
        let jsonFlag = reader.popFlag("--json")
        if jsonFlag { return jsonImplies }
        if let formatRaw {
            return OutputFormat(rawValue: formatRaw) ?? defaultFormat
        }
        return defaultFormat
    }
}
