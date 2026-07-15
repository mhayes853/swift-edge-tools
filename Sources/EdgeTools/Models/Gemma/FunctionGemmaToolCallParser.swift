// MARK: - FunctionGemmaToolCallParser

public struct FunctionGemmaToolCallParser: EdgeToolCallParser, Sendable {
  private var core = GemmaToolCallParserCore(
    syntax: GemmaToolCallSyntax(
      opener: "<start_function_call>",
      closer: "<end_function_call>",
      stringMarker: "<escape>",
      decodesMarkedValues: true
    )
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.core.accept(token: token)
  }
}
