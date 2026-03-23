import Foundation

struct ArgReader {
    private var args: [String]
    private var index: Int = 0

    init(_ args: [String]) { self.args = args }

    var isEmpty: Bool { index >= args.count }

    mutating func pop() -> String? {
        guard index < args.count else { return nil }
        defer { index += 1 }
        return args[index]
    }

    mutating func popValue(for flag: String) -> String? {
        guard index < args.count, args[index] == flag else { return nil }
        index += 1
        return pop()
    }

    mutating func popInt(for flag: String) -> Int? {
        guard let value = popValue(for: flag) else { return nil }
        return Int(value)
    }

    mutating func popFlag(_ flag: String) -> Bool {
        guard index < args.count, args[index] == flag else { return false }
        index += 1
        return true
    }
}

