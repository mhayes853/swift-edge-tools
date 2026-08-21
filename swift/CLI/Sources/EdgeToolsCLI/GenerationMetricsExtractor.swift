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
  public static let percentage = Self(precision: 1, suffix: "%")

  func string(from value: Double) -> String {
    self.numberString(from: value) + self.suffix
  }

  func numberString(from value: Double) -> String {
    String(format: "%.*f", self.precision, value)
  }
}

// MARK: - Confidence

/// The confidence group an engine reports, empty when the model scores neither.
func confidenceMetricGroup(from generation: EdgeToolsEngineGeneration) -> CLIMetricGroup? {
  var metrics = [CLIMetric]()
  if let confidence = generation.metrics.generationConfidence {
    metrics.append(
      CLIMetric(
        id: "generationConfidence",
        jsonKey: "generationConfidencePercentage",
        value: Double(confidence) * 100,
        format: .percentage,
        benchmarkLabel: "Confidence %",
        benchmarkAggregation: .distribution
      )
    )
  }
  if let probe = generation.metrics.probeConfidence {
    metrics.append(
      CLIMetric(
        id: "probeConfidence",
        jsonKey: "probeConfidencePercentage",
        label: "probe ",
        value: Double(probe) * 100,
        format: .percentage,
        benchmarkLabel: "Probe %",
        benchmarkAggregation: .distribution
      )
    )
  }
  return metrics.isEmpty ? nil : CLIMetricGroup(label: "Confidence", metrics: metrics)
}

// MARK: - StandardGenerationMetricsExtractor

public struct StandardGenerationMetricsExtractor: GenerationMetricsExtractor {
  public init() {}

  public func extract(from generation: EdgeToolsEngineGeneration) -> CLIGenerationMetrics {
    var groups = [CLIMetricGroup]()
    var prefillMetrics = [CLIMetric]()
    if let tokens = generation.metrics.prefillTokens {
      prefillMetrics.append(
        CLIMetric(id: "prefillTokens", jsonKey: "prefillTokens", value: Double(tokens), format: .tokenCount)
      )
    }
    if let duration = generation.metrics.prefillDuration {
      prefillMetrics.append(
        CLIMetric(
          id: "prefillMilliseconds",
          jsonKey: "prefillMilliseconds",
          value: duration.milliseconds,
          format: .milliseconds
        )
      )
    }
    if let rate = generation.metrics.prefillTokensPerSecond {
      prefillMetrics.append(
        CLIMetric(
          id: "prefillTokensPerSecond",
          jsonKey: "prefillTokensPerSecond",
          value: rate,
          format: .tokensPerSecond,
          benchmarkLabel: "Prefill tok/s",
          benchmarkAggregation: .distribution
        )
      )
    }
    if !prefillMetrics.isEmpty {
      groups.append(CLIMetricGroup(label: "Prefill", metrics: prefillMetrics))
    }

    var decodeMetrics = [CLIMetric]()
    if let tokens = generation.metrics.decodeTokens {
      decodeMetrics.append(
        CLIMetric(id: "decodeTokens", jsonKey: "decodeTokens", value: Double(tokens), format: .tokenCount)
      )
    }
    if let duration = generation.metrics.decodeDuration {
      decodeMetrics.append(
        CLIMetric(
          id: "decodeMilliseconds",
          jsonKey: "decodeMilliseconds",
          value: duration.milliseconds,
          format: .milliseconds
        )
      )
    }
    if let rate = generation.metrics.decodeTokensPerSecond {
      decodeMetrics.append(
        CLIMetric(
          id: "decodeTokensPerSecond",
          jsonKey: "decodeTokensPerSecond",
          value: rate,
          format: .tokensPerSecond,
          benchmarkLabel: "Decode tok/s",
          benchmarkAggregation: .distribution
        )
      )
    }
    if let duration = generation.metrics.durationToFirstToken {
      decodeMetrics.append(
        CLIMetric(
          id: "timeToFirstTokenMilliseconds",
          jsonKey: "timeToFirstTokenMilliseconds",
          label: "TTFT ",
          value: duration.milliseconds,
          format: .milliseconds,
          benchmarkLabel: "TTFT ms",
          benchmarkAggregation: .distribution
        )
      )
    }
    if !decodeMetrics.isEmpty {
      groups.append(CLIMetricGroup(label: "Decode", metrics: decodeMetrics))
    }

    if let confidence = confidenceMetricGroup(from: generation) {
      groups.append(confidence)
    }
    return CLIGenerationMetrics(groups: groups)
  }
}

// MARK: - Needle2GenerationMetricsExtractor

public struct Needle2GenerationMetricsExtractor: GenerationMetricsExtractor {
  public init() {}

  public func extract(from generation: EdgeToolsEngineGeneration) -> CLIGenerationMetrics {
    var groups = [CLIMetricGroup]()
    if let rate = generation.metrics.prefillTokensPerSecond {
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
    if let rate = generation.metrics.decodeTokensPerSecond {
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
    if let peakRAM = generation.metrics.needle2PeakRAMMegabytes {
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
    if let confidence = confidenceMetricGroup(from: generation) {
      groups.append(confidence)
    }
    return CLIGenerationMetrics(groups: groups)
  }
}
