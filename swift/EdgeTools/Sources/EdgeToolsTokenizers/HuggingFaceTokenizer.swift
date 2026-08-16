import EdgeToolsCore

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if FoundationEssentials
  import _EdgeToolsFoundation
#endif

#if HuggingFaceTokenizers && canImport(CTokenizers)
  import CTokenizers
#endif

#if HuggingFaceTokenizers && FoundationEssentials && canImport(CTokenizers)
  import OrderedCollections
#endif

// MARK: - HuggingFaceTokenizer

#if HuggingFaceTokenizers && FoundationEssentials && canImport(CTokenizers)
  public final class HuggingFaceTokenizer: @unchecked Sendable, EdgeToolsTokenizer {
    private let handle: OpaquePointer
    private let configuration: HuggingFaceTokenizerConfiguration

    public let unk: EdgeToolsToken?
    public let bos: EdgeToolsToken?
    public let eos: EdgeToolsToken?
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
      self.unk = nativeSpecialToken(parsed.unknownToken, handle: handle)
      self.bos = nativeSpecialToken(parsed.bosToken, handle: handle)
      self.eos = nativeSpecialToken(parsed.eosToken, handle: handle)
    }

    deinit { hf_tokenizer_destroy(self.handle) }

    public func encode(text: String) -> [EdgeToolsToken] {
      self.encode(text: text, addSpecialTokens: true)
    }

    public func decode(tokens: [EdgeToolsToken.ID]) -> String {
      self.decode(tokens: tokens, skipSpecialTokens: false)
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [EdgeToolsToken] {
      let ids = (try? nativeEncode(self.handle, text, addSpecialTokens: addSpecialTokens)) ?? []
      return self.tokens(forIds: ids).compactMap { $0 }
    }

    public func decode(tokens: [EdgeToolsToken.ID], skipSpecialTokens: Bool) -> String {
      (try? nativeDecode(self.handle, tokens, skipSpecialTokens: skipSpecialTokens)) ?? ""
    }

    public func tokens(forIds ids: [EdgeToolsToken.ID]) -> [EdgeToolsToken?] {
      ids.map { id in
        ((try? nativeIDToToken(self.handle, id)) ?? nil).map { EdgeToolsToken(id: id, stringValue: $0) }
      }
    }

    public func tokens(forTexts texts: [String]) -> [EdgeToolsToken?] {
      texts.map { text in
        ((try? nativeTokenToID(self.handle, text)) ?? nil)
          .map { EdgeToolsToken(id: $0, stringValue: text) }
      }
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
    ) throws -> [EdgeToolsToken] {
      let ids = try nativeEncode(
        self.handle,
        self.renderChatTemplate(
          messages: messages,
          tools: tools,
          addGenerationPrompt: addGenerationPrompt,
          additionalContext: additionalContext
        ),
        addSpecialTokens: false
      )
      return self.tokens(forIds: ids).compactMap { $0 }
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
        if let eosTokenId = self.eos?.id {
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

    extension XGRTokenizerInfo {
      public static func huggingFace(
        encodedVocabulary: [String],
        backendJSON: String,
        modelVocabularySize: Int? = nil,
        stopTokenIDs: [Int] = []
      ) throws -> XGRTokenizerInfo {
        let metadata = try Self.metadata(huggingFaceBackendJSON: backendJSON)
        guard
          case .object(let decodedMetadata) = try EdgeToolsValue(json: Array(metadata.utf8)),
          case .integer(let vocabularyTypeValue) = decodedMetadata["vocab_type"],
          case .boolean(let addPrefixSpace) = decodedMetadata["add_prefix_space"]
        else {
          throw EdgeToolsTokenizerError.invalidBackendJSON
        }
        let vocabularyType: XGRVocabularyType
        switch vocabularyTypeValue {
        case 0: vocabularyType = .raw
        case 1: vocabularyType = .byteFallback
        case 2: vocabularyType = .byteLevel
        default: throw EdgeToolsTokenizerError.invalidBackendJSON
        }
        return try XGRTokenizerInfo(
          encodedVocabulary: encodedVocabulary,
          vocabularyType: vocabularyType,
          vocabularySize: modelVocabularySize ?? encodedVocabulary.count,
          stopTokenIDs: stopTokenIDs,
          addPrefixSpace: addPrefixSpace
        )
      }
    }
  #endif

  // MARK: - Hugging Face Tokenizer Configuration

  private struct HuggingFaceTokenizerConfiguration: Sendable {
    let bosToken: String?
    let eosToken: String?
    let unknownToken: String?
    let sepToken: String?
    let padToken: String?
    let clsToken: String?
    let maskToken: String?
    let additionalSpecialTokens: [String]
    private let chatTemplates: [NamedChatTemplate]

    init(data: Data?, chatTemplateOverride: String? = nil) throws {
      let source = try data.map { try JSONDecoder().decode(Source.self, from: $0) } ?? Source()
      self.bosToken = source.bosToken?.content
      self.eosToken = source.eosToken?.content
      self.unknownToken = source.unknownToken?.content
      self.sepToken = source.sepToken?.content
      self.padToken = source.padToken?.content
      self.clsToken = source.clsToken?.content
      self.maskToken = source.maskToken?.content
      self.additionalSpecialTokens = source.additionalSpecialTokens.compactMap(\.content)
      self.chatTemplates = (chatTemplateOverride.map { [Source.ChatTemplate.literal($0)] }
        ?? source.chatTemplate)
        .map { NamedChatTemplate(source: $0) }
    }

    func renderChatTemplate(
      messages: [EdgeToolsValue],
      tools: [EdgeToolsValue]?,
      addGenerationPrompt: Bool,
      additionalContext: [String: EdgeToolsValue]?
    ) throws -> String {
      let source = try self.selectedChatTemplate(tools: tools)
      var context = OrderedDictionary<String, EdgeToolsValue>()
      self.addSpecialTokens(to: &context)
      context["messages"] = .array(messages)
      context["add_generation_prompt"] = .boolean(addGenerationPrompt)
      if let tools {
        context["tools"] = .array(tools)
      }
      context.merge(additionalContext ?? [:]) { _, override in override }
      return try nativeRenderTemplate(source, context: .object(context))
    }

    private func selectedChatTemplate(tools: [EdgeToolsValue]?) throws -> String {
      let toolTemplate =
        tools?.isEmpty == false
        ? self.chatTemplates.first { $0.name == toolUseChatTemplateName }
        : nil
      guard let template = toolTemplate
        ?? self.chatTemplates.first(where: { $0.name == defaultChatTemplateName })
      else {
        throw EdgeToolsTokenizerError(
          code: .missingChatTemplate,
          message: self.chatTemplates.isEmpty
            ? "The Hugging Face tokenizer does not define a chat template."
            : "The Hugging Face tokenizer has no default chat template."
        )
      }
      return template.source
    }

    private func addSpecialTokens(to context: inout OrderedDictionary<String, EdgeToolsValue>) {
      if let bosToken {
        context["bos_token"] = .string(bosToken)
      }
      if let eosToken {
        context["eos_token"] = .string(eosToken)
      }
      if let unknownToken {
        context["unk_token"] = .string(unknownToken)
      }
      if let sepToken {
        context["sep_token"] = .string(sepToken)
      }
      if let padToken {
        context["pad_token"] = .string(padToken)
      }
      if let clsToken {
        context["cls_token"] = .string(clsToken)
      }
      if let maskToken {
        context["mask_token"] = .string(maskToken)
      }
      if !additionalSpecialTokens.isEmpty {
        context["additional_special_tokens"] = .array(
          additionalSpecialTokens.map(EdgeToolsValue.string)
        )
      }
    }
  }

  // MARK: - Chat Templates

  private let defaultChatTemplateName = "default"
  private let toolUseChatTemplateName = "tool_use"

  private struct NamedChatTemplate: Sendable {
    let name: String
    let source: String

    init(source: Source.ChatTemplate) {
      switch source {
      case .literal(let template):
        self.name = defaultChatTemplateName
        self.source = template
      case .named(let name, let template):
        self.name = name
        self.source = template
      }
    }
  }

  // MARK: - Configuration Source

  private struct Source: Decodable {
    struct Token: Decodable {
      let content: String

      init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
          self.content = value
        } else {
          self.content = try container.decode(Content.self).content
        }
      }
    }

    struct Content: Decodable {
      let content: String
    }

    enum ChatTemplate: Decodable {
      case literal(String)
      case named(name: String, template: String)

      private enum CodingKeys: String, CodingKey {
        case name
        case template
      }

      init(from decoder: any Decoder) throws {
        if let value = try? decoder.singleValueContainer().decode(String.self) {
          self = .literal(value)
          return
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self = .named(
          name: try container.decode(String.self, forKey: .name),
          template: try container.decode(String.self, forKey: .template)
        )
      }
    }

    let bosToken: Token?
    let eosToken: Token?
    let unknownToken: Token?
    let sepToken: Token?
    let padToken: Token?
    let clsToken: Token?
    let maskToken: Token?
    let additionalSpecialTokens: [Token]
    let chatTemplate: [ChatTemplate]

    private enum CodingKeys: String, CodingKey {
      case bosToken = "bos_token"
      case eosToken = "eos_token"
      case unknownToken = "unk_token"
      case sepToken = "sep_token"
      case padToken = "pad_token"
      case clsToken = "cls_token"
      case maskToken = "mask_token"
      case additionalSpecialTokens = "additional_special_tokens"
      case chatTemplate = "chat_template"
    }

    init() {
      self.bosToken = nil
      self.eosToken = nil
      self.unknownToken = nil
      self.sepToken = nil
      self.padToken = nil
      self.clsToken = nil
      self.maskToken = nil
      self.additionalSpecialTokens = []
      self.chatTemplate = []
    }

    init(from decoder: any Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      self.bosToken = try container.decodeIfPresent(Token.self, forKey: .bosToken)
      self.eosToken = try container.decodeIfPresent(Token.self, forKey: .eosToken)
      self.unknownToken = try container.decodeIfPresent(Token.self, forKey: .unknownToken)
      self.sepToken = try container.decodeIfPresent(Token.self, forKey: .sepToken)
      self.padToken = try container.decodeIfPresent(Token.self, forKey: .padToken)
      self.clsToken = try container.decodeIfPresent(Token.self, forKey: .clsToken)
      self.maskToken = try container.decodeIfPresent(Token.self, forKey: .maskToken)
      self.additionalSpecialTokens =
        try container.decodeIfPresent([Token].self, forKey: .additionalSpecialTokens) ?? []
      if let templates = try? container.decodeIfPresent([ChatTemplate].self, forKey: .chatTemplate) {
        self.chatTemplate = templates
      } else {
        self.chatTemplate =
          try container.decodeIfPresent(ChatTemplate.self, forKey: .chatTemplate).map { [$0] } ?? []
      }
    }
  }

  // MARK: - Native Tokenizer Calls

  private func nativeCreateTokenizer(_ json: UnsafeBufferPointer<UInt8>) throws -> OpaquePointer {
    var tokenizer: OpaquePointer?
    try nativeCall {
      hf_tokenizer_create(json.baseAddress, json.count, &tokenizer)
    }
    guard let tokenizer else {
      throw EdgeToolsTokenizerError(
        code: .nativeFailure,
        message: "The native tokenizer did not create a tokenizer handle."
      )
    }
    return tokenizer
  }

  private func nativeSpecialToken(_ text: String?, handle: OpaquePointer) -> EdgeToolsToken? {
    text.flatMap { text in
      ((try? nativeTokenToID(handle, text)) ?? nil).map { EdgeToolsToken(id: $0, stringValue: text) }
    }
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
          throw EdgeToolsTokenizerError(
            code: .nativeFailure,
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
      throw EdgeToolsTokenizerError(
        code: .nativeFailure,
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
        throw EdgeToolsTokenizerError(
          code: .nativeFailure,
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
    throw EdgeToolsTokenizerError(
      code: .nativeFailure,
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
      if values.allSatisfy({ $0 != nil }) {
        break
      }

      self.skipWhitespace()
      if self.consumeIfPresent(.jsonCloseObject) {
        break
      }
      try self.consume(.jsonComma)
      self.skipWhitespace()
    }

    return try zip(Self.metadataKeys, values)
      .compactMap { key, value in
        guard let value else { return nil }
        let bytes = self.buffer[value]
        let jsonValue = String(decoding: bytes, as: UTF8.self)
        guard jsonValue.utf8.elementsEqual(bytes) else {
          throw EdgeToolsTokenizerError.invalidBackendJSON
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
    throw EdgeToolsTokenizerError.invalidBackendJSON
  }

  private mutating func valueRange() throws -> Range<Int> {
    let start = self.index
    guard self.index < self.buffer.count else { throw EdgeToolsTokenizerError.invalidBackendJSON }

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
    guard let end, end > start else { throw EdgeToolsTokenizerError.invalidBackendJSON }
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
        guard endings.popLast() == byte else { throw EdgeToolsTokenizerError.invalidBackendJSON }
        if endings.isEmpty {
          return
        }
      default: break
      }
    }
    throw EdgeToolsTokenizerError.invalidBackendJSON
  }

  private mutating func skipWhitespace() {
    while self.index < self.buffer.count, self.buffer[self.index].isASCIIWhitespace {
      self.index += 1
    }
  }

  private mutating func consume(_ byte: UInt8) throws {
    guard self.consumeIfPresent(byte) else { throw EdgeToolsTokenizerError.invalidBackendJSON }
  }

  private mutating func consumeIfPresent(_ byte: UInt8) -> Bool {
    guard self.index < self.buffer.count, self.buffer[self.index] == byte else { return false }
    self.index += 1
    return true
  }
}
