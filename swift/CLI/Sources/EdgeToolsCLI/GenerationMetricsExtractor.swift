import EdgeTools
import Foundation

// MARK: - GenerationMetricsExtractor

public protocol GenerationMetricsExtractor: Sendable {
  func extract(from generation: EdgeToolsEngineGeneration) -> CLIGenerationMetrics
}

// MARK: - CLIGenerationMetrics

public struct CLIGenerationMetrics: Sendable {
  public var groups: [CLIMetricGroup]

  public init(groups: [CLIMetricGroup]) {
    self.groups = groups
  }

  public subscript(id: String) -> Double? {
    self.metrics.first(where: { $0.id == id })?.value
  }

  var metrics: [CLIMetric] {
    self.groups.flatMap(\.metrics)
  }
}

// MARK: - CLIMetricGroup

public struct CLIMetricGroup: Sendable {
  public var label: String
  public var metrics: [CLIMetric]

  public init(label: String, metrics: [CLIMetric]) {
    self.label = label
    self.metrics = metrics
  }
}

// MARK: - CLIMetric

public struct CLIMetric: Sendable {
  public var id: String
  public var jsonKey: String
  public var label: String?
  public var value: Double
  public var format: CLIMetricFormat
  public var benchmarkLabel: String?
  public var benchmarkAggregation: CLIBenchmarkAggregation?

  public init(
    id: String,
    jsonKey: String,
    label: String? = nil,
    value: Double,
    format: CLIMetricFormat,
    benchmarkLabel: String? = nil,
    benchmarkAggregation: CLIBenchmarkAggregation? = nil
  ) {
    self.id = id
    self.jsonKey = jsonKey
    self.label = label
    self.value = value
    self.format = format
    self.benchmarkLabel = benchmarkLabel
    self.benchmarkAggregation = benchmarkAggregation
  }

  var displayText: String {
    (self.label ?? "") + self.format.string(from: self.value)
  }
}

// MARK: - CLIBenchmarkAggregation

public enum CLIBenchmarkAggregation: Hashable, Sendable {
  case distribution
  case maximum
}

// MARK: - CLIMetricFormat

public struct CLIMetricFormat: Hashable, Sendable {
  public var precision: Int
  public var suffix: String

  public init(precision: Int, suffix: String = "") {
    self.precision = precision
    self.suffix = suffix
  }

  public static let tokenCount = Self(precision: 0, suffix: " tok")
  public static let milliseconds = Self(precision: 0, suffix: "ms")
  public static let tokensPerSecond = Self(precision: 1, suffix: " tok/s")
  public static let megabytes = Self(precision: 1, suffix: " MB")

  func string(from value: Double) -> String {
    self.numberString(from: value) + self.suffix
  }

  func numberString(from value: Double) -> String {
    String(format: "%.*f", self.precision, value)
  }
}

// MARK: - StandardGenerationMetricsExtractor

public struct StandardGenerationMetricsExtractor: GenerationMetricsExtractor {
  public init() {}

  public func extract(from generation: EdgeToolsEngineGeneration) -> CLIGenerationMetrics {
    CLIGenerationMetrics(groups: [
      CLIMetricGroup(
        label: "Prefill",
        metrics: [
          CLIMetric(
            id: "prefillTokens",
            jsonKey: "prefillTokens",
            value: Double(generation.prefillMetrics.tokens),
            format: .tokenCount
          ),
          CLIMetric(
            id: "prefillMilliseconds",
            jsonKey: "prefillMilliseconds",
            value: generation.prefillMetrics.duration.milliseconds,
            format: .milliseconds
          ),
          CLIMetric(
            id: "prefillTokensPerSecond",
            jsonKey: "prefillTokensPerSecond",
            value: generation.prefillMetrics.tokensPerSecond,
            format: .tokensPerSecond,
            benchmarkLabel: "Prefill tok/s",
            benchmarkAggregation: .distribution
          )
        ]
      ),
      CLIMetricGroup(
        label: "Decode",
        metrics: [
          CLIMetric(
            id: "decodeTokens",
            jsonKey: "decodeTokens",
            value: Double(generation.decodeMetrics.tokens),
            format: .tokenCount
          ),
          CLIMetric(
            id: "decodeMilliseconds",
            jsonKey: "decodeMilliseconds",
            value: generation.decodeMetrics.duration.milliseconds,
            format: .milliseconds
          ),
          CLIMetric(
            id: "decodeTokensPerSecond",
            jsonKey: "decodeTokensPerSecond",
            value: generation.decodeMetrics.tokensPerSecond,
            format: .tokensPerSecond,
            benchmarkLabel: "Decode tok/s",
            benchmarkAggregation: .distribution
          ),
          CLIMetric(
            id: "timeToFirstTokenMilliseconds",
            jsonKey: "timeToFirstTokenMilliseconds",
            label: "TTFT ",
            value: generation.decodeMetrics.durationToFirstToken.milliseconds,
            format: .milliseconds,
            benchmarkLabel: "TTFT ms",
            benchmarkAggregation: .distribution
          )
        ]
      )
    ])
  }
}

// MARK: - Needle2GenerationMetricsExtractor

public struct Needle2GenerationMetricsExtractor: GenerationMetricsExtractor {
  public init() {}

  public func extract(from generation: EdgeToolsEngineGeneration) -> CLIGenerationMetrics {
    var groups = [CLIMetricGroup]()
    if let rate = generation.metadata.needle2PrefillTokensPerSecond {
      groups.append(
        CLIMetricGroup(
          label: "Prefill",
          metrics: [
            CLIMetric(
              id: "prefillTokensPerSecond",
              jsonKey: "prefillTokensPerSecond",
              value: rate,
              format: .tokensPerSecond,
              benchmarkLabel: "Prefill tok/s",
              benchmarkAggregation: .distribution
            )
          ]
        )
      )
    }
    if let rate = generation.metadata.needle2DecodeTokensPerSecond {
      groups.append(
        CLIMetricGroup(
          label: "Decode",
          metrics: [
            CLIMetric(
              id: "decodeTokensPerSecond",
              jsonKey: "decodeTokensPerSecond",
              value: rate,
              format: .tokensPerSecond,
              benchmarkLabel: "Decode tok/s",
              benchmarkAggregation: .distribution
            )
          ]
        )
      )
    }
    if let peakRAM = generation.metadata.needle2PeakRAMMegabytes {
      groups.append(
        CLIMetricGroup(
          label: "RAM",
          metrics: [
            CLIMetric(
              id: "needle2PeakRAMMegabytes",
              jsonKey: "needle2PeakRAMMegabytes",
              label: "peak ",
              value: peakRAM,
              format: .megabytes,
              benchmarkLabel: "Peak RAM",
              benchmarkAggregation: .maximum
            )
          ]
        )
      )
    }
    return CLIGenerationMetrics(groups: groups)
  }
}
