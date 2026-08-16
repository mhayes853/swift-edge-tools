import EdgeToolsCore

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if FoundationEssentials
  import _EdgeToolsFoundation
#endif

// MARK: - EdgeToolsTokenizer

public protocol EdgeToolsTokenizer: Sendable {
  var eos: EdgeToolsToken? { borrowing get }

  func encode(text: String) -> [EdgeToolsToken]
  func decode(tokens: [EdgeToolsToken.ID]) -> String
  func tokens(forIds ids: [EdgeToolsToken.ID]) -> [EdgeToolsToken?]
  func tokens(forTexts texts: [String]) -> [EdgeToolsToken?]
}

extension EdgeToolsTokenizer {
  public func token(forId id: EdgeToolsToken.ID) -> EdgeToolsToken? {
    self.tokens(forIds: [id])[0]
  }

  public func token(forText text: String) -> EdgeToolsToken? {
    self.tokens(forTexts: [text])[0]
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
  ) throws -> [EdgeToolsToken]
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
