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
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidTokenizerInfo = Self(rawValue: "invalid-tokenizer-info")
      public static let invalidHuggingFaceMetadata = Self(rawValue: "invalid-hugging-face-metadata")
      public static let invalidJSONSchemaConfiguration = Self(
        rawValue: "invalid-json-schema-configuration"
      )
      public static let invalidCompilerConfiguration = Self(
        rawValue: "invalid-compiler-configuration"
      )
      public static let invalidMatcherConfiguration = Self(
        rawValue: "invalid-matcher-configuration"
      )
      public static let invalidRepetitionRange = Self(rawValue: "invalid-repetition-range")
      public static let invalidToolInvocationRange = Self(rawValue: "invalid-tool-invocation-range")
      public static let emptyToolCollection = Self(rawValue: "empty-tool-collection")
      public static let unsupportedToolSchema = Self(rawValue: "unsupported-tool-schema")
      public static let incompatibleTokenizerVocabulary = Self(
        rawValue: "incompatible-tokenizer-vocabulary"
      )
      public static let invalidNeedleTokenizer = Self(rawValue: "invalid-needle-tokenizer")
      public static let xgrammarFailure = Self(rawValue: "xgrammar-failure")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }

    static let invalidHuggingFaceMetadata = Self(
      code: .invalidHuggingFaceMetadata,
      message: "Invalid Hugging Face tokenizer metadata."
    )
    static let invalidRepetitionRange = Self(
      code: .invalidRepetitionRange,
      message: "Invalid XGrammar repetition range."
    )
    static let invalidToolInvocationRange = Self(
      code: .invalidToolInvocationRange,
      message: "Tool invocation ranges cannot have a negative lower bound."
    )
    static let unsupportedToolSchema = Self(
      code: .unsupportedToolSchema,
      message: "The tool definition has an unsupported schema."
    )

    static func xgrammarFailure(message: String) -> Self {
      Self(code: .xgrammarFailure, message: message)
    }
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
        throw XGRError(
          code: .invalidTokenizerInfo,
          message: "Invalid XGrammar tokenizer information."
        )
      }
      let vocabularySize = try vocabularySize.map {
        try xgrammarInt32(
          $0,
          error: XGRError(
            code: .invalidTokenizerInfo,
            message: "Invalid XGrammar tokenizer information."
          )
        )
      }
      let stopTokenIDs = try stopTokenIDs.map {
        try xgrammarInt32(
          $0,
          error: XGRError(
            code: .invalidTokenizerInfo,
            message: "Invalid XGrammar tokenizer information."
          )
        )
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
      modelVocabularySize: Int? = nil,
      stopTokenIDs: [Int] = []
    ) throws -> XGRTokenizerInfo {
      let metadata = try Self.metadata(huggingFaceBackendJSON: backendJSON)
      guard
        case .object(let decodedMetadata) = try EdgeToolsJSONDecoder().decode(
          EdgeToolsValue.self,
          from: Array(metadata.utf8)
        ),
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
        vocabularySize: modelVocabularySize ?? encodedVocabulary.count,
        stopTokenIDs: stopTokenIDs,
        addPrefixSpace: addPrefixSpace
      )
    }

    public func serializedJSON() throws -> String {
      try xgrammarString { xgrammar_tokenizer_info_serialize_json(self.handle, $0, $1) }
    }
  }

  // MARK: - XGRGrammar

  /// An immutable XGrammar grammar.
  ///
  /// `@unchecked Sendable` is safe because XGrammar grammar handles are immutable after
  /// construction; this reference type owns and destroys its handle exactly once.
  public final class XGRGrammar: @unchecked Sendable {
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

    public static func ebnf(_ ebnf: String, rootRuleName: String = "root") throws -> XGRGrammar {
      let handle = try ebnf.withCString { ebnf in
        try rootRuleName.withCString {
          try xgrammarRequiredHandle(xgrammar_grammar_init_ebnf(ebnf, $0))
        }
      }
      return Self(handle: handle)
    }

    public static func regex(_ regex: String) throws -> XGRGrammar {
      let handle = try regex.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_init_regex($0))
      }
      return Self(handle: handle)
    }

    public static func lark(_ lark: String) throws -> XGRGrammar {
      let handle = try lark.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_init_lark($0))
      }
      return Self(handle: handle)
    }

    public static func structuralTagJSON(_ structuralTagJSON: String) throws -> XGRGrammar {
      let handle = try structuralTagJSON.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_init_structural_tag($0))
      }
      return Self(handle: handle)
    }

    public static func literal(_ literal: String) throws -> XGRGrammar {
      try Self.ebnf("root ::= \"\(Self.escapeEBNFLiteral(literal))\"")
    }

    public static func jsonSchema(
      _ jsonSchema: String,
      configuration: JSONSchemaConfiguration = JSONSchemaConfiguration()
    ) throws -> XGRGrammar {
      guard configuration.indent.map({ $0 >= 0 }) ?? true,
        configuration.maximumWhitespaceCount.map({ $0 >= 0 }) ?? true
      else {
        throw XGRError(
          code: .invalidJSONSchemaConfiguration,
          message: "Invalid XGrammar JSON Schema configuration."
        )
      }
      let indent = try configuration.indent.map {
        try xgrammarInt32(
          $0,
          error: XGRError(
            code: .invalidJSONSchemaConfiguration,
            message: "Invalid XGrammar JSON Schema configuration."
          )
        )
      }
      let maximumWhitespaceCount = try configuration.maximumWhitespaceCount.map {
        try xgrammarInt32(
          $0,
          error: XGRError(
            code: .invalidJSONSchemaConfiguration,
            message: "Invalid XGrammar JSON Schema configuration."
          )
        )
      }
      let handle = try jsonSchema.withCString { schema in
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
      return Self(handle: handle)
    }

    public static func schema(_ schema: EdgeToolsGenerationSchema) -> XGRGrammar {
      guard let grammar = try? Self.jsonSchema(schema.orderedKeyEncoded()) else {
        preconditionFailure("EdgeTools generation schemas must produce a valid XGrammar grammar.")
      }
      return grammar
    }

    public static func schema(_ type: (some EdgeToolsGenerable).Type) -> XGRGrammar {
      Self.schema(type.edgeToolsGenerationSchema)
    }

    public static var universal: XGRGrammar {
      guard let grammar = try? Self.structuralTagJSON(#"{"type":"structural_tag","format":{"type":"any_text"}}"#) else {
        preconditionFailure("XGrammar must support the any_text structural tag.")
      }
      return grammar
    }

    public static func serializedJSON(_ serializedJSON: String) throws -> XGRGrammar {
      let handle = try serializedJSON.withCString {
        try xgrammarRequiredHandle(xgrammar_grammar_deserialize_json($0))
      }
      return Self(handle: handle)
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

  // MARK: - XGRGenerationConstraint

  /// Selects the grammar used to constrain an engine generation.
  public enum XGRGenerationConstraint: Sendable {
    /// Allows arbitrary text output.
    case unconstrained

    /// Uses the supplied grammar directly.
    case grammar(XGRGrammar)

    /// Uses the engine's model-specific tool-call grammar.
    case tools(
      range: GrammarToolCallRange = .unbounded(minimum: 0),
      grammar: (@Sendable (XGRGrammar) throws -> XGRGrammar)? = nil
    )

    /// The default tool-call constraint.
    public static let tools = Self.tools()

    /// Constrains output to a value described by an EdgeTools generation schema.
    public static func schema(_ type: (some EdgeToolsGenerable).Type) -> Self {
      .grammar(.schema(type))
    }

    public var toolCallRange: GrammarToolCallRange? {
      if case .tools(let range, _) = self { range } else { nil }
    }

    /// Resolves this constraint using an engine's model-specific tool-call grammar.
    public func grammar(using toolsGrammar: XGRGrammar) throws -> XGRGrammar {
      switch self {
      case .unconstrained:
        .universal
      case .grammar(let grammar):
        grammar
      case .tools(_, let transform):
        try transform?(toolsGrammar) ?? toolsGrammar
      }
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
        error: XGRError(
          code: .invalidCompilerConfiguration,
          message: "Invalid XGrammar compiler configuration."
        )
      )
      guard maxThreads > 0 else {
        throw XGRError(
          code: .invalidCompilerConfiguration,
          message: "Invalid XGrammar compiler configuration."
        )
      }
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
        error: XGRError(
          code: .invalidMatcherConfiguration,
          message: "Invalid XGrammar matcher configuration."
        )
      )
      let overrideStopTokenIDs = try overrideStopTokenIDs.map {
        try xgrammarInt32(
          $0,
          error: XGRError(
            code: .invalidMatcherConfiguration,
            message: "Invalid XGrammar matcher configuration."
          )
        )
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
      throw XGRError.xgrammarFailure(message: String(cString: xgrammar_last_error_message()))
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
      throw XGRError.xgrammarFailure(message: String(cString: xgrammar_last_error_message()))
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

// MARK: - Grammar Combinators

#if XGrammar
  public func concatenate(
    _ lhs: borrowing XGRGrammar,
    _ rhs: borrowing XGRGrammar
  ) throws -> XGRGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try xgrammarRequiredHandle(xgrammar_grammar_concatenate($0.baseAddress, $0.count))
    }
    return XGRGrammar(handle: handle)
  }

  public func union(
    _ lhs: borrowing XGRGrammar,
    _ rhs: borrowing XGRGrammar
  ) throws -> XGRGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try xgrammarRequiredHandle(xgrammar_grammar_union($0.baseAddress, $0.count))
    }
    return XGRGrammar(handle: handle)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGRGrammar,
    exactly count: Int
  ) throws -> XGRGrammar {
    try repeatGrammar(grammar, count...count)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGRGrammar,
    _ range: ClosedRange<Int>
  ) throws -> XGRGrammar {
    guard range.lowerBound >= 0 else { throw XGRError.invalidRepetitionRange }
    let lowerBound = try xgrammarInt32(range.lowerBound, error: .invalidRepetitionRange)
    let upperBound = try xgrammarInt32(range.upperBound, error: .invalidRepetitionRange)
    let handle = try xgrammarRequiredHandle(
      xgrammar_grammar_repeat(grammar.handle, lowerBound, upperBound)
    )
    return XGRGrammar(handle: handle)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGRGrammar,
    _ range: PartialRangeFrom<Int>
  ) throws -> XGRGrammar {
    guard range.lowerBound >= 0 else { throw XGRError.invalidRepetitionRange }
    let lowerBound = try xgrammarInt32(range.lowerBound, error: .invalidRepetitionRange)
    let handle = try xgrammarRequiredHandle(
      xgrammar_grammar_repeat(grammar.handle, lowerBound, -1)
    )
    return XGRGrammar(handle: handle)
  }
#endif
