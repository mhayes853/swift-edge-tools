#if canImport(Tokenizers)
  import Tokenizers

  struct StreamingDetokenizer {
    private static let replacementCharacter = "\u{fffd}"

    private let tokenizer: any Tokenizers.Tokenizer
    private(set) var tokenIds = [EdgeToolsToken.ID]()
    private(set) var streamedResponse = ""

    init(tokenizer: any Tokenizers.Tokenizer) {
      self.tokenizer = tokenizer
    }

    mutating func decode(tokenId: EdgeToolsToken.ID) -> String {
      self.tokenIds.append(tokenId)
      let decodedResponse = self.tokenizer.decode(tokens: self.tokenIds)
      guard decodedResponse.hasPrefix(self.streamedResponse) else {
        self.streamedResponse = decodedResponse
        return decodedResponse
      }

      let startIndex = decodedResponse.index(
        decodedResponse.startIndex,
        offsetBy: self.streamedResponse.count
      )
      let tokenString = String(decodedResponse[startIndex...])
      guard !tokenString.hasSuffix(Self.replacementCharacter) else { return "" }
      self.streamedResponse = decodedResponse
      return tokenString
    }
  }
#endif
