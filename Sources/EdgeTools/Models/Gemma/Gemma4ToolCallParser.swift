// MARK: - Gemma4ToolCallParser

public struct Gemma4ToolCallParser: EdgeToolCallParser, Sendable {
  private var core = GemmaToolCallParserCore(
    syntax: GemmaToolCallSyntax(
      opener: "<|tool_call>",
      closer: "<tool_call|>",
      stringMarker: "<|\"|>",
      decodesMarkedValues: false
    )
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.core.accept(token: token)
  }
}
