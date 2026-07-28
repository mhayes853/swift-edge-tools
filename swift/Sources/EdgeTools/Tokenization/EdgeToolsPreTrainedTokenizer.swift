#if XGrammar
  import EdgeToolsXGrammar
#endif

#if System
  import SystemPackage
#endif

#if Foundation
  package import _EdgeToolsFoundation
#endif

#if Transformers
  import Tokenizers
#endif

// MARK: - EdgeToolsPreTrainedTokenizer

#if Transformers
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

  #if XGrammar
    extension EdgeToolsPreTrainedTokenizer: EdgeToolsXGRTokenizer {
      public func tokenizerInfo(modelVocabularySize: Int? = nil) throws -> XGRTokenizerInfo {
        var vocabulary = [String]()
        while let token = self.convertIdToToken(vocabulary.count) {
          vocabulary.append(token)
        }
        let vocabularySize = max(modelVocabularySize ?? 0, vocabulary.count)
        return try XGRTokenizerInfo.huggingFace(
          encodedVocabulary: vocabulary,
          backendJSON: self.backendJSON,
          modelVocabularySize: vocabularySize,
          stopTokenIDs: self.eosTokenId.map { [$0] } ?? []
        )
      }
    }
  #endif
#endif

// MARK: - HF Backend JSON

#if System
  package func loadHuggingFaceBackendJSON(from tokenizerPath: FilePath) throws -> String {
    try withFileBytes(at: tokenizerPath) { try huggingFaceBackendJSON(from: $0) }
  }
#endif

#if Foundation
  package func loadHuggingFaceBackendJSON(from tokenizerURL: URL) throws -> String {
    try Data(contentsOf: tokenizerURL).withUnsafeBytes { buffer in
      try huggingFaceBackendJSON(from: buffer.bindMemory(to: UInt8.self))
    }
  }
#endif

package func huggingFaceBackendJSON(from bytes: UnsafeBufferPointer<UInt8>) throws -> String {
  var scanner = HuggingFaceBackendJSONScanner(buffer: bytes)
  return "{\(try scanner.metadataFields().joined(separator: ","))}"
}

private struct HuggingFaceBackendJSONScanner {
  private static let metadataKeys = ["decoder", "normalizer", "pre_tokenizer"]

  let buffer: UnsafeBufferPointer<UInt8>
  var index = 0

  mutating func metadataFields() throws -> [String] {
    self.skipWhitespace()
    try self.consume(.jsonOpenObject)
    self.skipWhitespace()

    var values = [Range<Int>?](repeating: nil, count: Self.metadataKeys.count)
    while !self.consumeIfPresent(.jsonCloseObject) {
      let key = try self.stringRange()
      self.skipWhitespace()
      try self.consume(.jsonColon)
      self.skipWhitespace()

      let value = try self.valueRange()
      if let metadataIndex = Self.metadataKeys.firstIndex(where: {
        self.buffer[key].elementsEqual("\"\($0)\"".utf8)
      }) {
        values[metadataIndex] = value
      }
      if values.allSatisfy({ $0 != nil }) { break }

      self.skipWhitespace()
      if self.consumeIfPresent(.jsonCloseObject) { break }
      try self.consume(.jsonComma)
      self.skipWhitespace()
    }

    return try zip(Self.metadataKeys, values)
      .compactMap { key, value in
        guard let value else { return nil }
        let bytes = self.buffer[value]
        let jsonValue = String(decoding: bytes, as: UTF8.self)
        guard jsonValue.utf8.elementsEqual(bytes) else {
          throw EdgeToolsError.invalidHuggingFaceBackendJSON
        }
        return "\"\(key)\":\(jsonValue)"
      }
  }

  private mutating func stringRange() throws -> Range<Int> {
    let start = self.index
    try self.consume(.jsonQuote)
    var escaped = false
    while self.index < self.buffer.count {
      let byte = self.buffer[self.index]
      self.index += 1
      if byte == .jsonQuote, !escaped { return start..<self.index }
      escaped = byte == .jsonEscape && !escaped
    }
    throw EdgeToolsError.invalidHuggingFaceBackendJSON
  }

  private mutating func valueRange() throws -> Range<Int> {
    let start = self.index
    guard self.index < self.buffer.count else { throw EdgeToolsError.invalidHuggingFaceBackendJSON }

    switch self.buffer[self.index] {
    case .jsonQuote: _ = try self.stringRange()
    case .jsonOpenObject, .jsonOpenArray: try self.container()
    default:
      while self.index < self.buffer.count,
        ![.jsonComma, .jsonCloseObject].contains(self.buffer[self.index])
      {
        self.index += 1
      }
    }

    let end = self.buffer[start..<self.index].lastIndex(where: { !$0.isASCIIWhitespace })
      .map { $0 + 1 }
    guard let end, end > start else { throw EdgeToolsError.invalidHuggingFaceBackendJSON }
    return start..<end
  }

  private mutating func container() throws {
    var endings = [UInt8]()
    while self.index < self.buffer.count {
      let byte = self.buffer[self.index]
      if byte == .jsonQuote {
        _ = try self.stringRange()
        continue
      }

      self.index += 1
      switch byte {
      case .jsonOpenObject: endings.append(.jsonCloseObject)
      case .jsonOpenArray: endings.append(.jsonCloseArray)
      case .jsonCloseObject, .jsonCloseArray:
        guard endings.popLast() == byte else { throw EdgeToolsError.invalidHuggingFaceBackendJSON }
        if endings.isEmpty { return }
      default: break
      }
    }
    throw EdgeToolsError.invalidHuggingFaceBackendJSON
  }

  private mutating func skipWhitespace() {
    while self.index < self.buffer.count, self.buffer[self.index].isASCIIWhitespace {
      self.index += 1
    }
  }

  private mutating func consume(_ byte: UInt8) throws {
    guard self.consumeIfPresent(byte) else { throw EdgeToolsError.invalidHuggingFaceBackendJSON }
  }

  private mutating func consumeIfPresent(_ byte: UInt8) -> Bool {
    guard self.index < self.buffer.count, self.buffer[self.index] == byte else { return false }
    self.index += 1
    return true
  }
}
