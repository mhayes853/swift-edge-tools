public struct EdgeToolsGenerationChannel: Sendable {
  public var onToken: (@Sendable (EdgeToolsToken) -> Void)?
  public var onToolCall: (@Sendable (EdgeRawToolCall) -> Void)?

  public init(
    onToken: (@Sendable (EdgeToolsToken) -> Void)? = nil,
    onToolCall: (@Sendable (EdgeRawToolCall) -> Void)? = nil
  ) {
    self.onToken = onToken
    self.onToolCall = onToolCall
  }

  public func emit(token: EdgeToolsToken) {
    self.onToken?(token)
  }

  public func emit(toolCall: EdgeRawToolCall) {
    self.onToolCall?(toolCall)
  }
}
