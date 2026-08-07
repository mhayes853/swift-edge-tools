import EdgeTools
import Foundation

// MARK: - RunReport

public struct RunReport: Encodable {
  public struct ToolCall: Encodable {
    public let name: String
    public let arguments: EdgeToolsValue
  }

  public struct Metrics: Encodable {
    public let load: Duration
    public let endToEnd: Duration
    public let prefill: EdgeToolsPrefillMetrics
    public let decode: EdgeToolsDecodeMetrics
    public let peakResident: MemoryByteCount
    public let peakGPU: MemoryByteCount

    enum CodingKeys: String, CodingKey {
      case loadMilliseconds
      case endToEndMilliseconds
      case timeToFirstTokenMilliseconds
      case prefillTokens
      case prefillMilliseconds
      case prefillTokensPerSecond
      case decodeTokens
      case decodeMilliseconds
      case decodeTokensPerSecond
      case peakResidentBytes
      case peakGPUBytes
    }

    public func encode(to encoder: any Encoder) throws {
      var container = encoder.container(keyedBy: CodingKeys.self)
      try container.encode(self.load.milliseconds, forKey: .loadMilliseconds)
      try container.encode(self.endToEnd.milliseconds, forKey: .endToEndMilliseconds)
      try container.encode(
        self.decode.durationToFirstToken.milliseconds,
        forKey: .timeToFirstTokenMilliseconds
      )
      try container.encode(self.prefill.tokens, forKey: .prefillTokens)
      try container.encode(self.prefill.duration.milliseconds, forKey: .prefillMilliseconds)
      try container.encode(self.prefill.tokensPerSecond, forKey: .prefillTokensPerSecond)
      try container.encode(self.decode.tokens, forKey: .decodeTokens)
      try container.encode(self.decode.duration.milliseconds, forKey: .decodeMilliseconds)
      try container.encode(self.decode.tokensPerSecond, forKey: .decodeTokensPerSecond)
      try container.encode(self.peakResident, forKey: .peakResidentBytes)
      try container.encode(self.peakGPU, forKey: .peakGPUBytes)
    }
  }

  public let model: String
  public let engine: String
  public let response: String
  public let wasStopped: Bool
  public let toolCalls: [ToolCall]
  public let metrics: Metrics
}

// MARK: - Rendering

extension RunReport {
  public func displayText(includingResponse: Bool) -> String {
    var lines = [String]()
    if includingResponse {
      lines.append(self.response)
    }
    if !self.toolCalls.isEmpty {
      lines.append("\nTool calls")
      for (offset, call) in self.toolCalls.enumerated() {
        lines.append("  \(offset + 1). \(call.name)")
        for line in prettyJSON(call.arguments).split(separator: "\n") {
          lines.append("     \(line)")
        }
      }
    }
    let metrics = self.metrics
    lines.append("")
    lines.append("Model   \(self.model)")
    lines.append("Engine  \(self.engine)\(self.wasStopped ? "  (stopped early)" : "")")
    lines.append("Load    \(metrics.load.displayText)")
    lines.append(
      "Memory  \(metrics.peakResident.displayText) peak RSS"
        + (metrics.peakGPU.isEmpty ? "" : " · \(metrics.peakGPU.displayText) peak GPU")
    )
    lines.append(
      "Prefill \(metrics.prefill.tokens) tok  \(metrics.prefill.duration.displayText)  "
        + rateText(metrics.prefill.tokensPerSecond)
    )
    lines.append(
      "Decode  \(metrics.decode.tokens) tok  \(metrics.decode.duration.displayText)  "
        + rateText(metrics.decode.tokensPerSecond)
        + "  TTFT \(metrics.decode.durationToFirstToken.displayText)"
    )
    lines.append("E2E     \(metrics.endToEnd.displayText) (excludes load)")
    return lines.joined(separator: "\n")
  }

  public func jsonText() throws -> String {
    try encodedJSON(self)
  }
}

func rateText(_ tokensPerSecond: Double) -> String {
  guard tokensPerSecond.isFinite else { return "-" }
  return String(format: "%.1f tok/s", tokensPerSecond)
}

func encodedJSON(_ value: some Encodable) throws -> String {
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
  return String(decoding: try encoder.encode(value), as: UTF8.self)
}

private func prettyJSON(_ value: EdgeToolsValue) -> String {
  (try? encodedJSON(value)) ?? "<unencodable>"
}
