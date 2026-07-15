public protocol EdgeToolCallParser {
  init()

  mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall?
}
