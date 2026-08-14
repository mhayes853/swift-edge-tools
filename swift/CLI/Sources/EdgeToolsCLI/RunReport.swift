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
    public let generation: CLIGenerationMetrics
    public let peakResident: MemoryByteCount
    public let peakGPU: MemoryByteCount

    enum CodingKeys: String, CodingKey {
      case loadMilliseconds
      case endToEndMilliseconds
      case peakResidentBytes
      case peakGPUBytes
    }

    public func encode(to encoder: any Encoder) throws {
      var container = encoder.container(keyedBy: DynamicCodingKey.self)
      try container.encode(
        self.load.milliseconds,
        forKey: DynamicCodingKey(CodingKeys.loadMilliseconds.rawValue)
      )
      try container.encode(
        self.endToEnd.milliseconds,
        forKey: DynamicCodingKey(CodingKeys.endToEndMilliseconds.rawValue)
      )
      for metric in self.generation.metrics {
        try container.encode(metric.value, forKey: DynamicCodingKey(metric.jsonKey))
      }
      try container.encode(
        self.peakResident,
        forKey: DynamicCodingKey(CodingKeys.peakResidentBytes.rawValue)
      )
      try container.encode(
        self.peakGPU,
        forKey: DynamicCodingKey(CodingKeys.peakGPUBytes.rawValue)
      )
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
        for line in call.arguments.prettyJSONText.split(separator: "\n") {
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
      contentsOf: metrics.generation.groups.map {
        $0.label.rightPadded(to: 8) + $0.metrics.map(\.displayText).joined(separator: "  ")
      }
    )
    lines.append("E2E     \(metrics.endToEnd.displayText) (excludes load)")
    return lines.joined(separator: "\n")
  }

  public func jsonText() throws -> String {
    try self.encodedJSON()
  }
}
