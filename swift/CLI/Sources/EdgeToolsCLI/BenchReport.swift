import Foundation

// MARK: - BenchReport

public struct BenchReport: Sendable {
  public let model: String
  public let engine: String
  public let runs: Int
  public let warmup: Int
  public let samples: [BenchSample]
  public let peakResident: MemoryByteCount
  public let peakGPU: MemoryByteCount

  public var prefillRates: Distribution {
    Distribution(self.samples.map(\.prefill.tokensPerSecond))
  }

  public var decodeRates: Distribution {
    Distribution(self.samples.map(\.decode.tokensPerSecond))
  }

  public var timesToFirstToken: Distribution {
    Distribution(self.samples.map(\.decode.durationToFirstToken.milliseconds))
  }

  public var endToEndTimes: Distribution {
    Distribution(self.samples.map(\.endToEnd.milliseconds))
  }

  public var runsWithToolCalls: Int {
    self.samples.count(where: \.madeToolCalls)
  }
}

// MARK: - Distribution

public struct Distribution: Encodable, Equatable, Sendable {
  public let median: Double
  public let p95: Double
  public let min: Double
  public let max: Double

  public init(_ values: [Double]) {
    self.median = percentile(values, 0.5)
    self.p95 = percentile(values, 0.95)
    self.min = values.min() ?? 0
    self.max = values.max() ?? 0
  }
}

func percentile(_ values: [Double], _ fraction: Double) -> Double {
  guard !values.isEmpty else { return 0 }
  let sorted = values.sorted()
  let index = Int((Double(sorted.count - 1) * fraction).rounded())
  return sorted[index]
}

// MARK: - Rendering

extension BenchReport {
  public func displayText() -> String {
    var lines = [
      "\(self.model) · \(self.engine) · \(self.runs) runs (\(self.warmup) warmup)",
      "",
      "                   median       p95       min       max",
      row("Prefill tok/s", self.prefillRates, format: "%.1f"),
      row("Decode tok/s", self.decodeRates, format: "%.1f"),
      row("TTFT ms", self.timesToFirstToken, format: "%.0f"),
      row("E2E ms", self.endToEndTimes, format: "%.0f"),
      "",
      "Peak RSS          \(self.peakResident.displayText)"
    ]
    if !self.peakGPU.isEmpty {
      lines.append("Peak GPU          \(self.peakGPU.displayText)")
    }
    lines.append(
      "Tool calls        \(self.runsWithToolCalls)/\(self.runs) runs produced calls"
    )
    return lines.joined(separator: "\n")
  }

  public func jsonText() throws -> String {
    try encodedJSON(
      Payload(
        model: self.model,
        engine: self.engine,
        runs: self.runs,
        warmup: self.warmup,
        prefillTokensPerSecond: self.prefillRates,
        decodeTokensPerSecond: self.decodeRates,
        timeToFirstTokenMilliseconds: self.timesToFirstToken,
        endToEndMilliseconds: self.endToEndTimes,
        peakResidentBytes: self.peakResident,
        peakGPUBytes: self.peakGPU,
        runsWithToolCalls: self.runsWithToolCalls
      )
    )
  }

  private struct Payload: Encodable {
    public let model: String
    public let engine: String
    public let runs: Int
    public let warmup: Int
    let prefillTokensPerSecond: Distribution
    let decodeTokensPerSecond: Distribution
    let timeToFirstTokenMilliseconds: Distribution
    let endToEndMilliseconds: Distribution
    let peakResidentBytes: MemoryByteCount
    let peakGPUBytes: MemoryByteCount
    let runsWithToolCalls: Int
  }
}

private func row(_ label: String, _ distribution: Distribution, format: String) -> String {
  let cells = [distribution.median, distribution.p95, distribution.min, distribution.max]
    .map { String(format: format, $0).leftPadded(to: 10) }
  return label.rightPadded(to: 17) + cells.joined()
}

extension String {
  fileprivate func leftPadded(to length: Int) -> String {
    String(repeating: " ", count: Swift.max(0, length - self.count)) + self
  }

  fileprivate func rightPadded(to length: Int) -> String {
    self + String(repeating: " ", count: Swift.max(0, length - self.count))
  }
}
