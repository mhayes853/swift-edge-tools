#if canImport(Tokenizers)
  import Tokenizers
#endif

// MARK: - NeedlePrompt

public struct NeedlePrompt: Hashable, Sendable {
  public var system: String
  public var user: String
  public var tools: [NeedleToolDefinition]

  public init(system: String, user: String, tools: [NeedleToolDefinition]) {
    self.system = system
    self.user = user
    self.tools = tools
  }

  public func formatted() -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema = String(
      decoding: tools.map { $0.normalized() }.needlePromptEncoded(),
      as: UTF8.self
    )
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }
}

// MARK: - Tokenized

#if canImport(Tokenizers)
  extension NeedlePrompt {
    public func tokenized(using tokenizer: some Tokenizer) -> [NeedleToken] {
      let tokenIds = tokenizer.encode(text: self.formatted())
      let tokens = tokenizer.convertIdsToTokens(tokenIds)
      return zip(tokenIds, tokens)
        .compactMap { (tokenId, token) in
          token.map { NeedleToken(id: tokenId, stringValue: $0) }
        }
    }
  }
#endif
