import EdgeTools
import Foundation

// MARK: - StreamPrinter

final class StreamPrinter: Sendable {
  private let mode: StreamOption
  private let start: ContinuousClock.Instant
  private let clock = ContinuousClock()

  init(mode: StreamOption, start: ContinuousClock.Instant) {
    self.mode = mode
    self.start = start
  }

  func token(_ token: EdgeToolsToken) {
    switch self.mode {
    case .tokens:
      output(token.stringValue, terminator: "")
    case .events:
      output("\(self.elapsed) token \(encodedJSONString(token.stringValue))")
    case .none:
      break
    }
  }

  func toolCall(_ call: EdgeRawToolCall) {
    guard self.mode == .events else { return }
    output("\(self.elapsed) tool-call \(call.name) \(compactJSON(call.arguments))")
  }

  func finish() {
    if self.mode == .tokens { output() }
  }

  private var elapsed: String {
    self.start.duration(to: self.clock.now).formattedDuration
      .padding(
        toLength: 8,
        withPad: " ",
        startingAt: 0
      )
  }
}

// MARK: - Run Report

func printRunReport(
  generation: EdgeToolsEngineGeneration,
  loaded: LoadedModel,
  metrics: RunMetrics,
  stream: StreamOption
) {
  if stream == .none {
    output(generation.response)
  }
  if !generation.toolCalls.isEmpty {
    output("\nTool calls")
    for (offset, call) in generation.toolCalls.enumerated() {
      output("  \(offset + 1). \(call.name)")
      for line in prettyJSON(call.arguments).split(separator: "\n") {
        output("     \(line)")
      }
    }
  }
  output()
  output(
    """
    Model   \(loaded.detection.model.displayName)
    Engine  \(loaded.engine.rawValue)\(generation.wasStopped ? "  (stopped early)" : "")
    Load    \(metrics.loadDuration.formattedDuration)
    Memory  \(formattedBytes(metrics.peakResidentBytes)) peak RSS\
    \(metrics.peakGPUBytes > 0 ? " · \(formattedBytes(metrics.peakGPUBytes)) peak GPU" : "")
    Prefill \(metrics.prefill.tokens) tok  \(metrics.prefill.duration.formattedDuration)  \
    \(formattedRate(metrics.prefill.tokensPerSecond))
    Decode  \(metrics.decode.tokens) tok  \(metrics.decode.duration.formattedDuration)  \
    \(formattedRate(metrics.decode.tokensPerSecond))  \
    TTFT \(metrics.decode.durationToFirstToken.formattedDuration)
    E2E     \(metrics.generationDuration.formattedDuration) (excludes load)
    """
  )
}

func formattedRate(_ tokensPerSecond: Double) -> String {
  guard tokensPerSecond.isFinite else { return "-" }
  return String(format: "%.1f tok/s", tokensPerSecond)
}

// MARK: - JSON Report

func runJSON(
  generation: EdgeToolsEngineGeneration,
  loaded: LoadedModel,
  metrics: RunMetrics
) throws -> String {
  let report = RunReport(
    model: loaded.detection.model.displayName,
    engine: loaded.engine.rawValue,
    response: generation.response,
    wasStopped: generation.wasStopped,
    toolCalls: generation.toolCalls.map {
      RunReport.ToolCall(name: $0.name, arguments: $0.arguments)
    },
    metrics: RunReport.Metrics(
      loadMilliseconds: metrics.loadDuration.milliseconds,
      endToEndMilliseconds: metrics.generationDuration.milliseconds,
      timeToFirstTokenMilliseconds: metrics.decode.durationToFirstToken.milliseconds,
      prefillTokens: metrics.prefill.tokens,
      prefillMilliseconds: metrics.prefill.duration.milliseconds,
      prefillTokensPerSecond: metrics.prefill.tokensPerSecond,
      decodeTokens: metrics.decode.tokens,
      decodeMilliseconds: metrics.decode.duration.milliseconds,
      decodeTokensPerSecond: metrics.decode.tokensPerSecond,
      peakResidentBytes: metrics.peakResidentBytes,
      peakGPUBytes: metrics.peakGPUBytes
    )
  )
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
  return String(decoding: try encoder.encode(report), as: UTF8.self)
}

// MARK: - RunReport

struct RunReport: Encodable {
  struct ToolCall: Encodable {
    let name: String
    let arguments: EdgeToolsValue
  }

  struct Metrics: Encodable {
    let loadMilliseconds: Double
    let endToEndMilliseconds: Double
    let timeToFirstTokenMilliseconds: Double
    let prefillTokens: Int
    let prefillMilliseconds: Double
    let prefillTokensPerSecond: Double
    let decodeTokens: Int
    let decodeMilliseconds: Double
    let decodeTokensPerSecond: Double
    let peakResidentBytes: Int
    let peakGPUBytes: Int
  }

  let model: String
  let engine: String
  let response: String
  let wasStopped: Bool
  let toolCalls: [ToolCall]
  let metrics: Metrics
}

private func prettyJSON(_ value: EdgeToolsValue) -> String {
  encoded(value, formatting: [.prettyPrinted, .withoutEscapingSlashes])
}

private func compactJSON(_ value: EdgeToolsValue) -> String {
  encoded(value, formatting: [.withoutEscapingSlashes])
}

private func encoded(
  _ value: EdgeToolsValue,
  formatting: JSONEncoder.OutputFormatting
) -> String {
  let encoder = JSONEncoder()
  encoder.outputFormatting = formatting
  guard let data = try? encoder.encode(value) else { return "<unencodable>" }
  return String(decoding: data, as: UTF8.self)
}

private func encodedJSONString(_ value: String) -> String {
  encoded(.string(value), formatting: [.withoutEscapingSlashes])
}
