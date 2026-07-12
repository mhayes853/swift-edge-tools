extension EdgeToolsTokenizer where Self: ~Copyable {
  public var toolsToken: String { "<tools>" }

  public var toolsTokenId: EdgeToolsToken.ID? {
    self.convertTokenToId(self.toolsToken)
  }

  public var toolCallToken: String { "<tool_call>" }

  public var toolCallTokenId: EdgeToolsToken.ID? {
    self.convertTokenToId(self.toolCallToken)
  }

  public func decode(
    tokens: [EdgeToolsToken.ID],
    skipSpecialTokens: Bool
  ) -> String {
    guard skipSpecialTokens else { return self.decode(tokens: tokens) }
    let specialTokenIDs = [
      self.unknownTokenId,
      self.bosTokenId,
      self.eosTokenId,
      self.toolsTokenId,
      self.toolCallTokenId
    ]
    .compactMap { $0 }
    return self.decode(tokens: tokens.filter { !specialTokenIDs.contains($0) })
  }
}
