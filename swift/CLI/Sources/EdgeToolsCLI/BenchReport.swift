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
    self.distribution(for: "prefillTokensPerSecond")
  }

  public var decodeRates: Distribution {
    self.distribution(for: "decodeTokensPerSecond")
  }

  public var timesToFirstToken: Distribution {
    self.distribution(for: "timeToFirstTokenMilliseconds")
  }

  public var endToEndTimes: Distribution {
    Distribution(self.samples.map(\.endToEnd.milliseconds))
  }

  public var runsWithToolCalls: Int {
    self.samples.count(where: \.madeToolCalls)
  }

  private var generationMetrics: [BenchMetric] {
    var seen = Set<String>()
    return self.samples
      .flatMap { $0.metrics.metrics }
      .compactMap { metric in
        guard let label = metric.benchmarkLabel,
          let aggregation = metric.benchmarkAggregation,
          seen.insert(metric.id).inserted
        else {
          return nil
        }
        let values = self.samples.compactMap { $0.metrics[metric.id] }
        return BenchMetric(
          jsonKey: metric.jsonKey,
          label: label,
          format: metric.format,
          aggregation: aggregation,
          values: values
        )
      }
  }

  private func distribution(for id: String) -> Distribution {
    Distribution(self.samples.compactMap { $0.metrics[id] })
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

private func percentile(_ values: [Double], _ fraction: Double) -> Double {
  guard !values.isEmpty else { return 0 }
  let sorted = values.sorted()
  let position = Double(sorted.count - 1) * fraction
  let lower = Int(position.rounded(.down))
  let upper = Int(position.rounded(.up))
  guard lower != upper else { return sorted[lower] }
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
}

// MARK: - BenchMetric

private struct BenchMetric {
  var jsonKey: String
  var label: String
  var format: CLIMetricFormat
  var aggregation: CLIBenchmarkAggregation
  var values: [Double]
}

// MARK: - Rendering

extension BenchReport {
  public func displayText() -> String {
    let metrics = self.generationMetrics
    let distributions = metrics.filter { $0.aggregation == .distribution }
    let maxima = metrics.filter { $0.aggregation == .maximum }
    var lines = [
      "\(self.model) · \(self.engine) · \(self.runs) runs (\(self.warmup) warmup)",
      "",
      "                   median       p95       min       max"
    ]
    lines.append(
      contentsOf: distributions.map {
        row($0.label, Distribution($0.values), format: $0.format)
      }
    )
    lines.append(row("E2E ms", self.endToEndTimes, format: .milliseconds))
    lines.append("")
    lines.append(
      contentsOf: maxima.compactMap { metric in
        metric.values.max()
          .map {
            metric.label.rightPadded(to: 18) + metric.format.string(from: $0)
          }
      }
    )
    lines.append("Peak RSS          \(self.peakResident.displayText)")
    if !self.peakGPU.isEmpty {
      lines.append("Peak GPU          \(self.peakGPU.displayText)")
    }
    lines.append(
      "Tool calls        \(self.runsWithToolCalls)/\(self.runs) runs produced calls"
    )
    return lines.joined(separator: "\n")
  }

  public func jsonText() throws -> String {
    let metrics = self.generationMetrics
    var payload: [String: JSONMetricValue] = [
      "endToEndMilliseconds": .distribution(self.endToEndTimes),
      "peakResidentBytes": .integer(self.peakResident.bytes),
      "peakGPUBytes": .integer(self.peakGPU.bytes)
    ]
    for metric in metrics {
      switch metric.aggregation {
      case .distribution:
        payload[metric.jsonKey] = .distribution(Distribution(metric.values))
      case .maximum:
        payload[metric.jsonKey] = .number(metric.values.max() ?? 0)
      }
    }
    return try Payload(
      model: self.model,
      engine: self.engine,
      runs: self.runs,
      warmup: self.warmup,
      metrics: payload,
      runsWithToolCalls: self.runsWithToolCalls
    )
    .encodedJSON()
  }

  private struct Payload: Encodable {
    let model: String
    let engine: String
    let runs: Int
    let warmup: Int
    let metrics: [String: JSONMetricValue]
    let runsWithToolCalls: Int

    func encode(to encoder: any Encoder) throws {
      var container = encoder.container(keyedBy: DynamicCodingKey.self)
      try container.encode(self.model, forKey: DynamicCodingKey("model"))
      try container.encode(self.engine, forKey: DynamicCodingKey("engine"))
      try container.encode(self.runs, forKey: DynamicCodingKey("runs"))
      try container.encode(self.warmup, forKey: DynamicCodingKey("warmup"))
      for (key, value) in self.metrics {
        try container.encode(value, forKey: DynamicCodingKey(key))
      }
      try container.encode(
        self.runsWithToolCalls,
        forKey: DynamicCodingKey("runsWithToolCalls")
      )
    }
  }
}

private enum JSONMetricValue: Encodable {
  case distribution(Distribution)
  case integer(Int)
  case number(Double)

  func encode(to encoder: any Encoder) throws {
    switch self {
    case .distribution(let distribution):
      try distribution.encode(to: encoder)
    case .integer(let integer):
      var container = encoder.singleValueContainer()
      try container.encode(integer)
    case .number(let number):
      var container = encoder.singleValueContainer()
      try container.encode(number)
    }
  }
}

private func row(
  _ label: String,
  _ distribution: Distribution,
  format: CLIMetricFormat
) -> String {
  let cells = [distribution.median, distribution.p95, distribution.min, distribution.max]
    .map { format.numberString(from: $0) }
    .map { $0.leftPadded(to: 10) }
  return label.rightPadded(to: 17) + cells.joined()
}
