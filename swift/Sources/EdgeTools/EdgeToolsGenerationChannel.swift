public struct EdgeToolsGenerationChannel {
  public var onToken: ((EdgeToolsToken) -> Void)?
  public var onToolCall: ((EdgeRawToolCall) -> Void)?

  public init(
    onToken: ((EdgeToolsToken) -> Void)? = nil,
    onToolCall: ((EdgeRawToolCall) -> Void)? = nil
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
