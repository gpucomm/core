import Foundation

struct StatsSummary {
    let count: Int
    let min: Double
    let max: Double
    let mean: Double
    let p50: Double
    let p95: Double
}

enum Stats {
    static func summarize(_ values: [Double]) -> StatsSummary {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let count = sorted.count
        let min = sorted.first!
        let max = sorted.last!
        let mean = sorted.reduce(0.0, +) / Double(count)
        let p50 = percentile(sorted, 0.50)
        let p95 = percentile(sorted, 0.95)
        return StatsSummary(count: count, min: min, max: max, mean: mean, p50: p50, p95: p95)
    }

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        if sorted.count == 1 { return sorted[0] }
        let clamped = max(0.0, min(1.0, p))
        let x = clamped * Double(sorted.count - 1)
        let lo = Int(floor(x))
        let hi = Int(ceil(x))
        if lo == hi { return sorted[lo] }
        let t = x - Double(lo)
        return sorted[lo] * (1.0 - t) + sorted[hi] * t
    }
}

