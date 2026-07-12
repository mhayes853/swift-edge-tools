struct StreamingDetokenizer {
  private static let replacementCharacter = "\u{fffd}"

  private(set) var tokenIds = [EdgeToolsToken.ID]()
  private(set) var streamedResponse = ""

  mutating func decode(
    tokenId: EdgeToolsToken.ID,
    using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) -> String {
    self.tokenIds.append(tokenId)
    let decodedResponse = tokenizer.decode(tokens: self.tokenIds)
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
