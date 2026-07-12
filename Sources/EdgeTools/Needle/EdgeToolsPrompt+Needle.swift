extension EdgeToolsPrompt {
  public func needleFormatted() -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema = String(
      decoding: self.tools.map { $0.needleNormalized() }.needlePromptEncoded(),
      as: UTF8.self
    )
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }

  public func needleTokenized(
    using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) -> [EdgeToolsToken] {
    let tokenIds = tokenizer.encode(text: self.needleFormatted())
    let tokens = tokenizer.convertIdsToTokens(tokenIds)
    return zip(tokenIds, tokens)
      .compactMap { (tokenId, token) in
        token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
      }
  }
}
