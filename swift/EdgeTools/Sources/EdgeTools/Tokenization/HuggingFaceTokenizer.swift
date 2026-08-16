#if XGrammar
  import EdgeToolsXGrammar
#endif

#if FoundationEssentials
  import _EdgeToolsFoundation
#endif

#if HuggingFaceTokenizers && canImport(CTokenizers)
  import CTokenizers
#endif

// MARK: - HuggingFaceTokenizer

#if HuggingFaceTokenizers && FoundationEssentials && canImport(CTokenizers)
  public final class HuggingFaceTokenizer: @unchecked Sendable, EdgeToolsTokenizer {
    private let handle: OpaquePointer
    private let configuration: HuggingFaceTokenizerConfiguration

    public let unknownToken: String?
    public let bosToken: String?
    public let eosToken: String?
    public let unknownTokenId: EdgeToolsToken.ID?
    public let bosTokenId: EdgeToolsToken.ID?
    public let eosTokenId: EdgeToolsToken.ID?
    public let backendJSON: String

    public init(
      tokenizerJSON: Data,
      configuration: Data? = nil,
      chatTemplate: String? = nil
    ) throws {
      let parsed = try HuggingFaceTokenizerConfiguration(
        data: configuration,
        chatTemplateOverride: chatTemplate
      )
      let handle = try tokenizerJSON.withUnsafeBytes { buffer in
        try nativeCreateTokenizer(buffer.bindMemory(to: UInt8.self))
      }

      self.configuration = parsed
      self.handle = handle
      self.backendJSON = try loadHuggingFaceBackendJSON(from: tokenizerJSON)
      self.unknownToken = parsed.unknownToken
      self.bosToken = parsed.bosToken
      self.eosToken = parsed.eosToken
      self.unknownTokenId = parsed.unknownToken.flatMap { try? nativeTokenToID(handle, $0) }
      self.bosTokenId = parsed.bosToken.flatMap { try? nativeTokenToID(handle, $0) }
      self.eosTokenId = parsed.eosToken.flatMap { try? nativeTokenToID(handle, $0) }
    }

    deinit { hf_tokenizer_destroy(self.handle) }

    public func encode(text: String) -> [EdgeToolsToken.ID] {
      self.encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokens: [EdgeToolsToken.ID]) -> String {
      self.decode(tokens: tokens, skipSpecialTokens: false)
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [EdgeToolsToken.ID] {
      (try? nativeEncode(self.handle, text, addSpecialTokens: addSpecialTokens)) ?? []
    }

    public func decode(tokens: [EdgeToolsToken.ID], skipSpecialTokens: Bool) -> String {
      (try? nativeDecode(self.handle, tokens, skipSpecialTokens: skipSpecialTokens)) ?? ""
    }

    public func convertTokensToIds(_ tokens: [String]) -> [EdgeToolsToken.ID?] {
      tokens.map { try? nativeTokenToID(self.handle, $0) }
    }

    public func convertIdsToTokens(_ ids: [EdgeToolsToken.ID]) -> [String?] {
      ids.map { try? nativeIDToToken(self.handle, $0) }
    }

    public func renderChatTemplate(
      messages: [EdgeToolsValue],
      tools: [EdgeToolsValue]?,
      addGenerationPrompt: Bool,
      additionalContext: [String: EdgeToolsValue]? = nil
    ) throws -> String {
      try self.configuration.renderChatTemplate(
        messages: messages,
        tools: tools,
        addGenerationPrompt: addGenerationPrompt,
        additionalContext: additionalContext
      )
    }

    public func applyChatTemplate(
      messages: [EdgeToolsValue],
      tools: [EdgeToolsValue]?,
      addGenerationPrompt: Bool,
      additionalContext: [String: EdgeToolsValue]? = nil
    ) throws -> [EdgeToolsToken.ID] {
      try nativeEncode(
        self.handle,
        self.renderChatTemplate(
          messages: messages,
          tools: tools,
          addGenerationPrompt: addGenerationPrompt,
          additionalContext: additionalContext
        ),
        addSpecialTokens: false
      )
    }
  }

  extension HuggingFaceTokenizer: EdgeToolsChatTokenizer {}

  #if XGrammar
    extension HuggingFaceTokenizer: XGRTokenizer {
      public func tokenizerInfo(
        modelVocabularySize: Int?,
        extraStopTokenIds: Set<EdgeToolsToken.ID>
      ) throws -> XGRTokenizerInfo {
        let vocabulary = try nativeVocabulary(self.handle)
        var stopTokenIds = extraStopTokenIds
        if let eosTokenId = self.eosTokenId {
          stopTokenIds.insert(eosTokenId)
        }
        return try XGRTokenizerInfo.huggingFace(
          encodedVocabulary: vocabulary,
          backendJSON: self.backendJSON,
          modelVocabularySize: max(modelVocabularySize ?? 0, vocabulary.count),
          stopTokenIDs: stopTokenIds.sorted()
        )
      }
    }
  #endif

  // MARK: - Native Tokenizer Calls

  private func nativeCreateTokenizer(_ json: UnsafeBufferPointer<UInt8>) throws -> OpaquePointer {
    var tokenizer: OpaquePointer?
    try nativeCall {
      hf_tokenizer_create(json.baseAddress, json.count, &tokenizer)
    }
    guard let tokenizer else {
      throw EdgeToolsError(
        code: .unsupportedTokenizer,
        message: "The native tokenizer did not create a tokenizer handle."
      )
    }
    return tokenizer
  }

  private func nativeEncode(
    _ tokenizer: OpaquePointer,
    _ text: String,
    addSpecialTokens: Bool
  ) throws -> [EdgeToolsToken.ID] {
    try Array(text.utf8).withUnsafeBufferPointer { text in
      try withNativeBuffer(capacity: text.count + 8) { ids, capacity, count in
        hf_tokenizer_encode(
          tokenizer,
          text.baseAddress,
          text.count,
          addSpecialTokens,
          ids,
          capacity,
          count
        )
      } transform: { ids in
        ids.map(EdgeToolsToken.ID.init)
      }
    }
  }

  private func nativeDecode(
    _ tokenizer: OpaquePointer,
    _ tokenIDs: [EdgeToolsToken.ID],
    skipSpecialTokens: Bool
  ) throws -> String {
    try tokenIDs.map(nativeTokenID).withUnsafeBufferPointer { tokenIDs in
      try withNativeBuffer(capacity: tokenIDs.count * 8 + 16) { text, capacity, count in
        hf_tokenizer_decode(
          tokenizer,
          tokenIDs.baseAddress,
          tokenIDs.count,
          skipSpecialTokens,
          text,
          capacity,
          count
        )
      } transform: { text in
        String(decoding: text, as: UTF8.self)
      }
    }
  }

  private func nativeTokenToID(
    _ tokenizer: OpaquePointer,
    _ token: String
  ) throws -> EdgeToolsToken.ID? {
    var tokenID: Int32 = 0
    var found = false
    try Array(token.utf8).withUnsafeBufferPointer { token -> Void in
      try nativeCall {
        hf_tokenizer_token_to_id(tokenizer, token.baseAddress, token.count, &tokenID, &found)
      }
    }
    return found ? EdgeToolsToken.ID(tokenID) : nil
  }

  private func nativeIDToToken(
    _ tokenizer: OpaquePointer,
    _ tokenID: EdgeToolsToken.ID
  ) throws -> String? {
    let tokenID = try nativeTokenID(tokenID)
    var found = false
    let token = try withNativeBuffer(capacity: 64) { token, capacity, count in
      hf_tokenizer_id_to_token(tokenizer, tokenID, token, capacity, count, &found)
    } transform: { token in
      String(decoding: token, as: UTF8.self)
    }
    return found ? token : nil
  }

  private func nativeVocabulary(_ tokenizer: OpaquePointer) throws -> [String] {
    var byteCount = 0
    var entryCount = 0
    try nativeCall {
      hf_tokenizer_vocabulary(tokenizer, nil, 0, &byteCount, nil, 0, &entryCount)
    }

    var storage = [UInt8](repeating: 0, count: byteCount)
    var sizes = [Int](repeating: 0, count: entryCount)
    try storage.withUnsafeMutableBufferPointer { storage in
      try sizes.withUnsafeMutableBufferPointer { sizes -> Void in
        let status = try nativeCall {
          hf_tokenizer_vocabulary(
            tokenizer,
            storage.baseAddress,
            storage.count,
            &byteCount,
            sizes.baseAddress,
            sizes.count,
            &entryCount
          )
        }
        guard status == HF_TOKENIZER_SUCCESS else {
          throw EdgeToolsError(
            code: .unsupportedTokenizer,
            message: "The native tokenizer vocabulary did not fit the size it reported."
          )
        }
      }
    }

    var offset = 0
    return sizes.map { size in
      defer { offset += size }
      return String(decoding: storage[offset..<(offset + size)], as: UTF8.self)
    }
  }

  func nativeRenderTemplate(_ source: String, context: EdgeToolsValue) throws -> String {
    try Array(source.utf8).withUnsafeBufferPointer { source in
      try Array(context.orderedJSONString().utf8).withUnsafeBufferPointer { context in
        let estimate = source.count + context.count + 4096
        return try withNativeBuffer(capacity: estimate) { text, capacity, count in
          hf_template_render(
            source.baseAddress,
            source.count,
            context.baseAddress,
            context.count,
            text,
            capacity,
            count
          )
        } transform: { text in
          String(decoding: text, as: UTF8.self)
        }
      }
    }
  }

  private func nativeTokenID(_ tokenID: EdgeToolsToken.ID) throws -> Int32 {
    guard let tokenID = Int32(exactly: tokenID) else {
      throw EdgeToolsError(
        code: .unsupportedTokenizer,
        message: "A token ID does not fit in the native tokenizer representation."
      )
    }
    return tokenID
  }

  private func withNativeBuffer<Element: FixedWidthInteger, Value>(
    capacity: Int,
    _ body: (UnsafeMutablePointer<Element>?, Int, UnsafeMutablePointer<Int>) -> Int32,
    transform: (UnsafeBufferPointer<Element>) -> Value
  ) throws -> Value {
    var capacity = capacity
    while true {
      var storage = [Element](repeating: 0, count: capacity)
      var required = 0
      let value = try storage.withUnsafeMutableBufferPointer { storage -> Value? in
        let status = try nativeCall { body(storage.baseAddress, storage.count, &required) }
        guard status == HF_TOKENIZER_SUCCESS else { return nil }
        return transform(UnsafeBufferPointer(start: storage.baseAddress, count: required))
      }
      if let value {
        return value
      }
      guard required > capacity else {
        throw EdgeToolsError(
          code: .unsupportedTokenizer,
          message: "The native tokenizer reported an inconsistent output size."
        )
      }
      capacity = required
    }
  }

  @discardableResult
  private func nativeCall(_ body: () -> Int32) throws -> Int32 {
    let status = body()
    guard status != HF_TOKENIZER_SUCCESS, status != HF_TOKENIZER_BUFFER_TOO_SMALL else {
      return status
    }
    throw EdgeToolsError(
      code: .unsupportedTokenizer,
      message: String(cString: hf_tokenizer_last_error_message())
    )
  }
#endif

// MARK: - HF Backend JSON

#if FoundationEssentials
  package func loadHuggingFaceBackendJSON(from tokenizerURL: URL) throws -> String {
    try loadHuggingFaceBackendJSON(from: Data(contentsOf: tokenizerURL))
  }

  package func loadHuggingFaceBackendJSON(from data: Data) throws -> String {
    try data.withUnsafeBytes { buffer in
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
      if byte == .jsonQuote, !escaped {
        return start..<self.index
      }
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
        if endings.isEmpty {
          return
        }
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
