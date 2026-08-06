#if XGrammar
  import EdgeToolsXGrammar
#endif

#if Foundation
  import _EdgeToolsFoundation
#endif

// MARK: - EdgeToolsTokenizer

public protocol EdgeToolsTokenizer: Sendable {
  var unknownTokenId: EdgeToolsToken.ID? { borrowing get }
  var bosTokenId: EdgeToolsToken.ID? { borrowing get }
  var eosTokenId: EdgeToolsToken.ID? { borrowing get }

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

// MARK: - EdgeToolsXGRTokenizer

#if XGrammar
  public protocol EdgeToolsXGRTokenizer: EdgeToolsTokenizer {
    func tokenizerInfo(modelVocabularySize: Int?) throws -> XGRTokenizerInfo
  }

  // MARK: - AnyEdgeToolsXGRTokenizer

  public struct AnyEdgeToolsXGRTokenizer: EdgeToolsXGRTokenizer {
    private let tokenizer: any EdgeToolsXGRTokenizer

    public init(_ tokenizer: any EdgeToolsXGRTokenizer) {
      self.tokenizer = tokenizer
    }

    public var unknownTokenId: EdgeToolsToken.ID? { self.tokenizer.unknownTokenId }
    public var bosTokenId: EdgeToolsToken.ID? { self.tokenizer.bosTokenId }
    public var eosTokenId: EdgeToolsToken.ID? { self.tokenizer.eosTokenId }

    public func encode(text: String) -> [EdgeToolsToken.ID] {
      self.tokenizer.encode(text: text)
    }

    public func decode(tokens: [EdgeToolsToken.ID]) -> String {
      self.tokenizer.decode(tokens: tokens)
    }

    public func convertTokensToIds(_ tokens: [String]) -> [EdgeToolsToken.ID?] {
      self.tokenizer.convertTokensToIds(tokens)
    }

    public func convertIdsToTokens(_ ids: [EdgeToolsToken.ID]) -> [String?] {
      self.tokenizer.convertIdsToTokens(ids)
    }

    public func tokenizerInfo(modelVocabularySize: Int?) throws -> XGRTokenizerInfo {
      try self.tokenizer.tokenizerInfo(modelVocabularySize: modelVocabularySize)
    }
  }
#endif
