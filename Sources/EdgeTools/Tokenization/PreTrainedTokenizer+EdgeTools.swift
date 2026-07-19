#if Transformers
  import Tokenizers

  public struct EdgeToolsPreTrainedTokenizer: EdgeToolsTokenizer {
    public let tokenizer: PreTrainedTokenizer
    public let backendJSON: String

    public init(tokenizer: PreTrainedTokenizer, backendJSON: String) {
      self.tokenizer = tokenizer
      self.backendJSON = backendJSON
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
  }
#endif
