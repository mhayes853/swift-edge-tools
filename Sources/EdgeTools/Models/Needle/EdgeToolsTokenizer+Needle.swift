extension EdgeToolsTokenizer where Self: ~Copyable {
  public var toolsToken: String { "<tools>" }

  public var toolsTokenId: EdgeToolsToken.ID? {
    self.convertTokenToId(self.toolsToken)
  }

  public var toolCallToken: String { "<tool_call>" }

  public var toolCallTokenId: EdgeToolsToken.ID? {
    self.convertTokenToId(self.toolCallToken)
  }
}
