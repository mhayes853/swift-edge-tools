extension EdgeToolsPrompt {
  public func needleFormatted() throws -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema = try self.tools.map { $0.needleNormalized() }.needlePromptEncoded()
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }

  public func needleTokenized(
    using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) throws -> [EdgeToolsToken] {
    let tokenIds = tokenizer.encode(text: try self.needleFormatted())
    let tokens = tokenizer.convertIdsToTokens(tokenIds)
    return zip(tokenIds, tokens)
      .compactMap { (tokenId, token) in
        token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
      }
  }
}
