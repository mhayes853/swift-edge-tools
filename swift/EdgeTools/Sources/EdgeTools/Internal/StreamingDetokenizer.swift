import EdgeToolsCore
import EdgeToolsTokenizers

struct StreamingDetokenizer {
  private static let replacementCharacter = "\u{fffd}"

  private(set) var tokenIds = [EdgeToolsToken.ID]()
  private(set) var streamedResponse = ""

  mutating func decode(
    tokenId: EdgeToolsToken.ID,
    using tokenizer: any EdgeToolsTokenizer
  ) -> String {
    self.tokenIds.append(tokenId)
    let decodedResponse = tokenizer.decode(tokens: self.tokenIds)
    let streamedBytes = self.streamedResponse.utf8
    let decodedBytes = decodedResponse.utf8
    guard decodedBytes.starts(with: streamedBytes) else {
      self.streamedResponse = decodedResponse
      return decodedResponse
    }

    let tokenString = String(decoding: decodedBytes.dropFirst(streamedBytes.count), as: UTF8.self)
    guard !tokenString.hasSuffix(Self.replacementCharacter) else { return "" }
    self.streamedResponse = decodedResponse
    return tokenString
  }
}
