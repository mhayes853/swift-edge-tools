import EdgeToolsCore

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if FoundationEssentials
  import _EdgeToolsFoundation
#endif

// MARK: - EdgeToolsTokenizer

public protocol EdgeToolsTokenizer: Sendable {
  var unknownTokenId: EdgeToolsToken.ID? { borrowing get }
  var bosTokenId: EdgeToolsToken.ID? { borrowing get }
  var eosTokenId: EdgeToolsToken.ID? { borrowing get }

  var unknownToken: String? { borrowing get }
  var bosToken: String? { borrowing get }
  var eosToken: String? { borrowing get }

  func encode(text: String) -> [EdgeToolsToken.ID]
  func decode(tokens: [EdgeToolsToken.ID]) -> String
  func convertTokensToIds(_ tokens: [String]) -> [EdgeToolsToken.ID?]
  func convertIdsToTokens(_ ids: [EdgeToolsToken.ID]) -> [String?]
}

extension EdgeToolsTokenizer {
  public func tokenize(text: String) -> [String] {
    self.encode(text: text).compactMap { self.convertIdToToken($0) }
  }

  public func convertTokenToId(_ token: String) -> EdgeToolsToken.ID? {
    self.convertTokensToIds([token])[0]
  }

  public func convertIdToToken(_ id: EdgeToolsToken.ID) -> String? {
    self.convertIdsToTokens([id])[0]
  }

  public var unknownToken: String? {
    guard let tokenID = self.unknownTokenId else { return nil }
    return self.convertIdToToken(tokenID)
  }

  public var bosToken: String? {
    guard let tokenID = self.bosTokenId else { return nil }
    return self.convertIdToToken(tokenID)
  }

  public var eosToken: String? {
    guard let tokenID = self.eosTokenId else { return nil }
    return self.convertIdToToken(tokenID)
  }
}

// MARK: - EdgeToolsChatTokenizer

public protocol EdgeToolsChatTokenizer: EdgeToolsTokenizer {
  func renderChatTemplate(
    messages: [EdgeToolsValue],
    tools: [EdgeToolsValue]?,
    addGenerationPrompt: Bool,
    additionalContext: [String: EdgeToolsValue]?
  ) throws -> String

  func applyChatTemplate(
    messages: [EdgeToolsValue],
    tools: [EdgeToolsValue]?,
    addGenerationPrompt: Bool,
    additionalContext: [String: EdgeToolsValue]?
  ) throws -> [EdgeToolsToken.ID]
}

// MARK: - XGRTokenizer

#if XGrammar
  public protocol XGRTokenizer: EdgeToolsTokenizer {
    func tokenizerInfo(
      modelVocabularySize: Int?,
      extraStopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> XGRTokenizerInfo
  }
#endif
