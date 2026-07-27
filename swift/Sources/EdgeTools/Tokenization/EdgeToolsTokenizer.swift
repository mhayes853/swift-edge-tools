#if System
  import SystemPackage
#endif

#if Foundation
  import Foundation
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
#endif
