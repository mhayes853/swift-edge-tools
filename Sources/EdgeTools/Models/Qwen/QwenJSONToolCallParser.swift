import Foundation

// MARK: - QwenJSONToolCallParser

public struct QwenJSONToolCallParser: EdgeToolCallParser, Sendable {
  private var block = IncrementalToolCallBlock(
    opener: "<tool_call>",
    closer: "</tool_call>"
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.block.append(token)
    while let payload = self.block.nextPayload(respectingJSONStringBoundaries: true) {
      if let call = try? JSONDecoder().decode(EdgeRawToolCall.self, from: payload) {
        return call
      }
    }
    return nil
  }
}

// MARK: - Qwen

public typealias Qwen3ToolCallParser = QwenJSONToolCallParser
