#if XGrammar
  import CXGrammar

  // MARK: - XGRVocabularyType

  public enum XGRVocabularyType: Hashable, Sendable {
    case raw
    case byteFallback
    case byteLevel

    public var rawValue: xgrammar_vocab_type_t {
      switch self {
      case .raw: xgrammar_vocab_type_raw
      case .byteFallback: xgrammar_vocab_type_byte_fallback
      case .byteLevel: xgrammar_vocab_type_byte_level
      }
    }
  }

  // MARK: - XGRError

  public struct XGRError: Error, Hashable, Sendable {
    public let message: String

    public init(message: String) {
      self.message = message
    }

    public static let invalidTokenizerInfo = Self(
      message: "Invalid XGrammar tokenizer information."
    )
    public static let invalidHuggingFaceMetadata = Self(
      message: "Invalid Hugging Face tokenizer metadata."
    )
    public static let invalidJSONSchemaConfiguration = Self(
      message: "Invalid XGrammar JSON Schema configuration."
    )
    public static let invalidCompilerConfiguration = Self(
      message: "Invalid XGrammar compiler configuration."
    )
    public static let invalidMatcherConfiguration = Self(
      message: "Invalid XGrammar matcher configuration."
    )
    public static let invalidRepetitionRange = Self(message: "Invalid XGrammar repetition range.")
    public static let emptyGrammarCollection = Self(message: "Expected at least one grammar.")
  }

  // MARK: - XGRTokenizerInfo

  public struct XGRTokenizerInfo: ~Copyable {
    public let handle: xgrammar_tokenizer_info_t

    public init(
      encodedVocabulary: [String],
      vocabularyType: XGRVocabularyType,
      vocabularySize: Int? = nil,
      stopTokenIDs: [Int] = [],
      addPrefixSpace: Bool = false
    ) throws {
      guard vocabularySize.map({ $0 >= 0 }) ?? true else {
        throw XGRError.invalidTokenizerInfo
      }
      let vocabularySize = try vocabularySize.map {
        try xgrammarInt32($0, error: .invalidTokenizerInfo)
      }
      let stopTokenIDs = try stopTokenIDs.map {
        try xgrammarInt32($0, error: .invalidTokenizerInfo)
      }
      self.handle = try withCopiedCStringPointerBuffer(encodedVocabulary) { vocabulary in
        try stopTokenIDs.withUnsafeBufferPointer { stopTokenIDs in
          try xgrammarRequiredHandle(
            xgrammar_tokenizer_info_init(
              vocabulary.baseAddress,
              vocabulary.count,
              vocabularyType.rawValue,
              vocabularySize ?? -1,
              stopTokenIDs.baseAddress,
              stopTokenIDs.count,
              addPrefixSpace.intValue(as: Int32.self)
            )
          )
        }
      }
    }

    public init(encodedVocabulary: [String], metadata: String) throws {
      self.handle = try withCopiedCStringPointerBuffer(encodedVocabulary) { vocabulary in
        try metadata.withCString {
          try xgrammarRequiredHandle(
            xgrammar_tokenizer_info_from_vocab_and_metadata(
              vocabulary.baseAddress,
              vocabulary.count,
              $0
            )
          )
        }
      }
    }

    public init(serializedJSON: String) throws {
      self.handle = try serializedJSON.withCString {
        try xgrammarRequiredHandle(xgrammar_tokenizer_info_deserialize_json($0))
      }
    }

    public init(handle: consuming xgrammar_tokenizer_info_t) {
      self.handle = consume handle
    }

    deinit { xgrammar_tokenizer_info_destroy(self.handle) }

    public static func metadata(huggingFaceBackendJSON: String) throws -> String {
      try huggingFaceBackendJSON.withCString { backendJSON in
        try xgrammarString { xgrammar_tokenizer_info_detect_metadata_from_hf(backendJSON, $0, $1) }
      }
    }

    public static func huggingFace(
      encodedVocabulary: [String],
      backendJSON: String,
      stopTokenIDs: [Int] = []
    ) throws -> XGRTokenizerInfo {
      let metadata = try Self.metadata(huggingFaceBackendJSON: backendJSON)
      guard case .object(let decodedMetadata) = try decodeEdgeToolsJSON(Array(metadata.utf8)),
        case .integer(let vocabularyTypeValue) = decodedMetadata["vocab_type"],
        case .boolean(let addPrefixSpace) = decodedMetadata["add_prefix_space"]
      else { throw XGRError.invalidHuggingFaceMetadata }
      let vocabularyType: XGRVocabularyType
      switch vocabularyTypeValue {
      case 0: vocabularyType = .raw
      case 1: vocabularyType = .byteFallback
      case 2: vocabularyType = .byteLevel
      default: throw XGRError.invalidHuggingFaceMetadata
      }
      return try XGRTokenizerInfo(
        encodedVocabulary: encodedVocabulary,
        vocabularyType: vocabularyType,
        stopTokenIDs: stopTokenIDs,
        addPrefixSpace: addPrefixSpace
      )
    }

    public func serializedJSON() throws -> String {
      try xgrammarString { xgrammar_tokenizer_info_serialize_json(self.handle, $0, $1) }
    }
  }

  // MARK: - XGRGrammar

  public struct XGRGrammar: ~Copyable {
    public struct JSONSchemaConfiguration: Hashable, Sendable {
      public struct Separators: Hashable, Sendable {
        public var comma: String
        public var colon: String
        public init(comma: String, colon: String) {
          self.comma = comma
          self.colon = colon
        }
      }

      public var anyWhitespace: Bool
      public var indent: Int?
      public var separators: Separators?
      public var isStrict: Bool
      public var maximumWhitespaceCount: Int?
      public var anyOrder: Bool

      public init(
        anyWhitespace: Bool = true,
        indent: Int? = nil,
        separators: Separators? = nil,
        isStrict: Bool = true,
        maximumWhitespaceCount: Int? = nil,
        anyOrder: Bool = false
      ) {
        self.anyWhitespace = anyWhitespace
        self.indent = indent
        self.separators = separators
        self.isStrict = isStrict
        self.maximumWhitespaceCount = maximumWhitespaceCount
        self.anyOrder = anyOrder
      }
    }

    public let handle: xgrammar_grammar_t

    public init(ebnf: String, rootRuleName: String = "root") throws {
      self.handle = try ebnf.withCString { ebnf in
        try rootRuleName.withCString {
          try xgrammarRequiredHandle(xgrammar_grammar_init_ebnf(ebnf, $0))
        }
      }
    }

    public init(regex: String) throws {
      self.handle = try regex.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_init_regex($0))
      }
    }

    public init(structuralTagJSON: String) throws {
      self.handle = try structuralTagJSON.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_init_structural_tag($0))
      }
    }

    public init(literal: String) throws {
      try self.init(ebnf: "root ::= \"\(Self.escapeEBNFLiteral(literal))\"")
    }

    public init(
      jsonSchema: String,
      configuration: JSONSchemaConfiguration = JSONSchemaConfiguration()
    ) throws {
      guard configuration.indent.map({ $0 >= 0 }) ?? true,
        configuration.maximumWhitespaceCount.map({ $0 >= 0 }) ?? true
      else { throw XGRError.invalidJSONSchemaConfiguration }
      let indent = try configuration.indent.map {
        try xgrammarInt32($0, error: .invalidJSONSchemaConfiguration)
      }
      let maximumWhitespaceCount = try configuration.maximumWhitespaceCount.map {
        try xgrammarInt32($0, error: .invalidJSONSchemaConfiguration)
      }
      self.handle = try jsonSchema.withCString { schema in
        if let separators = configuration.separators {
          return try separators.comma.withCString { comma in
            try separators.colon.withCString { colon in
              try xgrammarRequiredHandle(
                xgrammar_grammar_init_json_schema(
                  schema,
                  configuration.anyWhitespace.intValue(as: Int32.self),
                  indent ?? -1,
                  comma,
                  colon,
                  configuration.isStrict.intValue(as: Int32.self),
                  maximumWhitespaceCount ?? -1,
                  configuration.anyOrder.intValue(as: Int32.self)
                )
              )
            }
          }
        }
        return try xgrammarRequiredHandle(
          xgrammar_grammar_init_json_schema(
            schema,
            configuration.anyWhitespace.intValue(as: Int32.self),
            indent ?? -1,
            nil,
            nil,
            configuration.isStrict.intValue(as: Int32.self),
            maximumWhitespaceCount ?? -1,
            configuration.anyOrder.intValue(as: Int32.self)
          )
        )
      }
    }

    public init(serializedJSON: String) throws {
      self.handle = try serializedJSON.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_deserialize_json($0))
      }
    }

    public init(handle: consuming xgrammar_grammar_t) {
      self.handle = consume handle
    }

    deinit { xgrammar_grammar_destroy(self.handle) }

    public static func builtinJSONGrammar() -> XGRGrammar {
      XGRGrammar(handle: xgrammar_grammar_builtin_json()!)
    }

    public var ebnf: String {
      let capacity = xgrammar_grammar_ebnf(self.handle, nil, 0)
      return xgrammarString(
        { xgrammar_grammar_ebnf(self.handle, $0, $1) },
        capacity: capacity
      )
    }

    public func serializedJSON() throws -> String {
      try xgrammarString { xgrammar_grammar_serialize_json(self.handle, $0, $1) }
    }

    private static func escapeEBNFLiteral(_ literal: String) -> String {
      literal.reduce(into: "") { result, character in
        if character == "\\" || character == "\"" { result.append("\\") }
        result.append(character)
      }
    }

    public borrowing func concatenate(
      _ grammar: borrowing XGRGrammar
    ) throws -> XGRGrammar {
      try EdgeTools.concatenate(self, grammar)
    }

    public borrowing func union(_ grammar: borrowing XGRGrammar) throws -> XGRGrammar {
      try EdgeTools.union(self, grammar)
    }

    public borrowing func optional() throws -> XGRGrammar {
      try XGRGrammar(handle: xgrammarRequiredHandle(xgrammar_grammar_optional(self.handle)))
    }

    public borrowing func repeated(exactly count: Int) throws -> XGRGrammar {
      try repeatGrammar(self, exactly: count)
    }

    public borrowing func repeated(_ range: ClosedRange<Int>) throws -> XGRGrammar {
      try repeatGrammar(self, range)
    }

    public borrowing func repeated(_ range: PartialRangeFrom<Int>) throws -> XGRGrammar {
      try repeatGrammar(self, range)
    }
  }

  // MARK: - XGRCompiledGrammar

  public struct XGRCompiledGrammar: ~Copyable {
    public let handle: xgrammar_compiled_grammar_t

    public init(handle: consuming xgrammar_compiled_grammar_t) {
      self.handle = consume handle
    }

    public init(serializedJSON: String, tokenizerInfo: borrowing XGRTokenizerInfo) throws {
      self.handle = try serializedJSON.withCString {
        try xgrammarRequiredHandle(
          xgrammar_compiled_grammar_deserialize_json($0, tokenizerInfo.handle)
        )
      }
    }

    deinit { xgrammar_compiled_grammar_destroy(self.handle) }

    public var grammar: XGRGrammar {
      XGRGrammar(handle: xgrammar_compiled_grammar_grammar(self.handle)!)
    }

    public var tokenizerInfo: XGRTokenizerInfo {
      XGRTokenizerInfo(handle: xgrammar_compiled_grammar_tokenizer_info(self.handle)!)
    }

    public var memorySizeBytes: Int64 {
      xgrammar_compiled_grammar_memory_size_bytes(self.handle)
    }

    public func serializedJSON() throws -> String {
      try xgrammarString { xgrammar_compiled_grammar_serialize_json(self.handle, $0, $1) }
    }
  }

  // MARK: - XGRCompiler

  public struct XGRCompiler: ~Copyable {
    public let handle: xgrammar_compiler_t

    public init(
      tokenizerInfo: borrowing XGRTokenizerInfo,
      maxThreads: Int = 8,
      cacheEnabled: Bool = true,
      maxMemoryBytes: Int64 = -1
    ) throws {
      let maxThreads = try xgrammarInt32(
        maxThreads,
        error: .invalidCompilerConfiguration
      )
      guard maxThreads > 0 else { throw XGRError.invalidCompilerConfiguration }
      self.handle = try xgrammarRequiredHandle(
        xgrammar_compiler_init(
          tokenizerInfo.handle,
          maxThreads,
          cacheEnabled.intValue(as: Int32.self),
          maxMemoryBytes
        )
      )
    }

    public init(handle: consuming xgrammar_compiler_t) {
      self.handle = consume handle
    }

    deinit { xgrammar_compiler_destroy(self.handle) }

    public var cacheSizeBytes: Int64 {
      xgrammar_compiler_cache_size_bytes(self.handle)
    }

    public var cacheLimitBytes: Int64 {
      xgrammar_compiler_cache_limit_bytes(self.handle)
    }

    public func clearCache() {
      xgrammar_compiler_clear_cache(self.handle)
    }

    public borrowing func compile(
      _ grammar: borrowing XGRGrammar
    ) throws -> XGRCompiledGrammar {
      try XGRCompiledGrammar(
        handle: xgrammarRequiredHandle(
          xgrammar_compiler_compile_grammar(self.handle, grammar.handle)
        )
      )
    }
  }

  // MARK: - XGRMatcher

  public struct XGRMatcher: ~Copyable {
    public let handle: xgrammar_matcher_t

    public init(
      compiledGrammar: borrowing XGRCompiledGrammar,
      overrideStopTokenIDs: [Int] = [],
      terminateWithoutStopToken: Bool = false,
      maxRollbackTokens: Int = -1
    ) throws {
      let maxRollbackTokens = try xgrammarInt32(
        maxRollbackTokens,
        error: .invalidMatcherConfiguration
      )
      let overrideStopTokenIDs = try overrideStopTokenIDs.map {
        try xgrammarInt32($0, error: .invalidMatcherConfiguration)
      }

      self.handle = try overrideStopTokenIDs.withUnsafeBufferPointer {
        try xgrammarRequiredHandle(
          xgrammar_matcher_init(
            compiledGrammar.handle,
            $0.baseAddress,
            $0.count,
            terminateWithoutStopToken.intValue(as: Int32.self),
            maxRollbackTokens
          )
        )
      }
    }

    public init(handle: consuming xgrammar_matcher_t) {
      self.handle = consume handle
    }

    deinit {
      xgrammar_matcher_destroy(self.handle)
    }

    public var isCompleted: Bool {
      xgrammar_matcher_is_completed(self.handle).boolValue
    }

    public var isTerminated: Bool {
      xgrammar_matcher_is_terminated(self.handle).boolValue
    }

    public func bitmask() -> GrammarBitmask {
      var bitmask = GrammarBitmask(bitCount: Int(xgrammar_matcher_bit_count(self.handle)))
      _ = bitmask.storage.withUnsafeMutableBytes {
        xgrammar_matcher_bitmask(self.handle, $0.bindMemory(to: Int32.self).baseAddress)
      }
      return bitmask
    }

    @discardableResult
    public func accept(tokenId: Int) -> Bool {
      xgrammar_matcher_accept_token(self.handle, Int32(tokenId)).boolValue
    }

    @discardableResult
    public func accept(string: String) -> Bool {
      string.withCString { xgrammar_matcher_accept_string(self.handle, $0).boolValue }
    }

    public func rollback(_ tokenCount: Int = 1) {
      xgrammar_matcher_rollback(self.handle, Int32(tokenCount))
    }

    public func reset() {
      xgrammar_matcher_reset(self.handle)
    }

    public borrowing func fork() -> XGRMatcher {
      XGRMatcher(handle: xgrammar_matcher_fork(self.handle)!)
    }
  }

  // MARK: - Helpers

  func xgrammarRequiredHandle<Handle>(_ handle: Handle?) throws -> Handle {
    guard let handle else {
      throw XGRError(message: String(cString: xgrammar_last_error_message()))
    }
    return handle
  }

  func xgrammarInt32(_ value: Int, error: XGRError) throws -> Int32 {
    guard let value = Int32(exactly: value) else { throw error }
    return value
  }

  private func xgrammarString(
    _ body: (UnsafeMutablePointer<CChar>?, Int) -> Int
  ) throws -> String {
    let capacity = body(nil, 0)
    guard capacity > 0 else {
      throw XGRError(message: String(cString: xgrammar_last_error_message()))
    }
    return xgrammarString(body, capacity: capacity)
  }

  private func xgrammarString(
    _ body: (UnsafeMutablePointer<CChar>?, Int) -> Int,
    capacity: Int
  ) -> String {
    var buffer = [CChar](repeating: 0, count: capacity)
    _ = buffer.withUnsafeMutableBufferPointer { body($0.baseAddress, $0.count) }
    return buffer.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
  }
#endif
