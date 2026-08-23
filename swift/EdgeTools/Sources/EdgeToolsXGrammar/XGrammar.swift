import CXGrammar

// MARK: - XGRVocabularyType

/// An XGrammar vocabulary type.
@nonexhaustive
public enum XGRVocabularyType: RawRepresentable, Hashable, Sendable {
  /// Raw vocabulary tokens.
  case raw

  /// Vocabulary tokens that use byte fallback encoding.
  case byteFallback

  /// Byte-level vocabulary tokens.
  case byteLevel

  public init?(rawValue: xgrammar_vocab_type_t) {
    switch rawValue {
    case xgrammar_vocab_type_raw: self = .raw
    case xgrammar_vocab_type_byte_fallback: self = .byteFallback
    case xgrammar_vocab_type_byte_level: self = .byteLevel
    default: return nil
    }
  }

  public var rawValue: xgrammar_vocab_type_t {
    switch self {
    case .raw: xgrammar_vocab_type_raw
    case .byteFallback: xgrammar_vocab_type_byte_fallback
    case .byteLevel: xgrammar_vocab_type_byte_level
    }
  }
}

// MARK: - XGRError

/// An error thrown by XGrammar.
public struct XGRError: Error, Hashable, Sendable {
  public struct Code: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    /// Invalid ``XGRTokenizerInfo``.
    public static let invalidTokenizerInfo = Self(rawValue: "invalid-tokenizer-info")

    /// An invalid JSON Schema configuration.
    public static let invalidJSONSchemaConfiguration = Self(
      rawValue: "invalid-json-schema-configuration"
    )

    /// An invalid ``XGRCompiler`` configuration.
    public static let invalidCompilerConfiguration = Self(
      rawValue: "invalid-compiler-configuration"
    )

    /// An invalid ``XGRMatcher`` configuration.
    public static let invalidMatcherConfiguration = Self(
      rawValue: "invalid-matcher-configuration"
    )

    /// An invalid repetition range for an ``XGRGrammar``.
    public static let invalidRepetitionRange = Self(rawValue: "invalid-repetition-range")

    /// The ``xgrammarFailure`` value.
    public static let xgrammarFailure = Self(rawValue: "xgrammar-failure")
  }

  /// The ``Code`` value.
  public let code: Code

  /// A human readable message for this error.
  public let message: String

  /// Creates an error.
  ///
  /// - Parameters:
  ///   - code: The ``code`` argument.
  ///   - message: The ``message`` argument.
  public init(code: Code, message: String) {
    self.code = code
    self.message = message
  }

  static let invalidTokenizerInfo = Self(
    code: .invalidTokenizerInfo,
    message: "Invalid XGrammar tokenizer information."
  )
  static let invalidJSONSchemaConfiguration = Self(
    code: .invalidJSONSchemaConfiguration,
    message: "Invalid XGrammar JSON Schema configuration."
  )
  static let invalidCompilerConfiguration = Self(
    code: .invalidCompilerConfiguration,
    message: "Invalid XGrammar compiler configuration."
  )
  static let invalidMatcherConfiguration = Self(
    code: .invalidMatcherConfiguration,
    message: "Invalid XGrammar matcher configuration."
  )
  static let invalidRepetitionRange = Self(
    code: .invalidRepetitionRange,
    message: "Invalid XGrammar repetition range."
  )
}

// MARK: - XGRTokenizerInfo

/// Immutable tokenizer metadata used by XGrammar grammar construction and compilation.
public final class XGRTokenizerInfo: @unchecked Sendable {
  /// The underlying pointer.
  fileprivate let handle: xgrammar_tokenizer_info_t

  /// Creates tokenizer info.
  ///
  /// - Parameters:
  ///   - encodedVocabulary: The tokenizer vocabulary.
  ///   - vocabularyType: The ``XGRVocabularyType`` of the tokenizer.
  ///   - vocabularySize: The size of the tokenizer vocabulary.
  ///   - stopTokenIDs: The token ids to stop generation.
  ///   - addPrefixSpace: Whether or not a prefix space is needed.
  public init(
    encodedVocabulary: [String],
    vocabularyType: XGRVocabularyType,
    vocabularySize: Int? = nil,
    stopTokenIDs: [Int] = [],
    addPrefixSpace: Bool = false
  ) throws {
    guard vocabularySize.map({ $0 >= 0 }) ?? true else { throw XGRError.invalidTokenizerInfo }
    let vocabularySize = try vocabularySize.map {
      try Int32($0, error: .invalidTokenizerInfo)
    }
    let stopTokenIDs = try stopTokenIDs.map {
      try Int32($0, error: .invalidTokenizerInfo)
    }
    self.handle = try withCopiedCStringPointerBuffer(encodedVocabulary) { vocabulary in
      try stopTokenIDs.withUnsafeBufferPointer { stopTokenIDs in
        try require(
          xgrammar_tokenizer_info_init(
            vocabulary.baseAddress,
            vocabulary.count,
            vocabularyType.rawValue,
            vocabularySize ?? -1,
            stopTokenIDs.baseAddress,
            stopTokenIDs.count,
            addPrefixSpace.intValue
          )
        )
      }
    }
  }

  /// Creates tokenizer info from a metadata string.
  ///
  /// You can obtain a metadata string from a HuggingFace `tokenizer.json` file by calling
  /// ``metadata(huggingFaceBackendJSON:)``.
  ///
  /// - Parameters:
  ///   - encodedVocabulary: The ``encodedVocabulary`` argument.
  ///   - metadata: A metadata string.
  public init(encodedVocabulary: [String], metadata: String) throws {
    self.handle = try withCopiedCStringPointerBuffer(encodedVocabulary) { vocabulary in
      try metadata.withCString {
        try require(
          xgrammar_tokenizer_info_from_vocab_and_metadata(
            vocabulary.baseAddress,
            vocabulary.count,
            $0
          )
        )
      }
    }
  }

  /// Creates tokenizer info from a serialized JSON string.
  ///
  /// - Parameters:
  ///   - serializedJSON: The ``serializedJSON`` argument.
  public init(serializedJSON: String) throws {
    self.handle = try serializedJSON.withCString {
      try require(xgrammar_tokenizer_info_deserialize_json($0))
    }
  }

  /// Creates tokenizer info from a raw pointer.
  ///
  /// - Parameters:
  ///   - handle: The pointer.
  public init(handle: consuming xgrammar_tokenizer_info_t) {
    self.handle = consume handle
  }

  deinit { xgrammar_tokenizer_info_destroy(self.handle) }

  /// Creates a metadata string from the string contents of HuggingFace `tokenizer.json` file.
  ///
  /// - Parameters:
  ///   - huggingFaceBackendJSON: The HuggingFace `tokenizer.json` contents.
  /// - Returns: A metadata string that you can pass to ``init(encodedVocabulary:metadata:)``.
  public static func metadata(huggingFaceBackendJSON: String) throws -> String {
    try huggingFaceBackendJSON.withCString { backendJSON in
      try xgrammarString { xgrammar_tokenizer_info_detect_metadata_from_hf(backendJSON, $0, $1) }
    }
  }

  /// Serializes this tokenizer info to a JSON string.
  public func serializedJSON() throws -> String {
    try xgrammarString { xgrammar_tokenizer_info_serialize_json(self.handle, $0, $1) }
  }

  /// Creates an independent tokenizer info value.
  public func copy() throws -> XGRTokenizerInfo {
    XGRTokenizerInfo(handle: try require(xgrammar_tokenizer_info_copy(self.handle)))
  }

  /// Calls `body` with the underlying tokenizer info pointer.
  ///
  /// The pointer is only valid for the duration of `body`.
  ///
  /// - Parameters:
  ///   - body: A closure that uses the pointer.
  /// - Returns: The result of `body`.
  public func withUnsafePointer<R>(
    _ body: (xgrammar_tokenizer_info_t) throws -> R
  ) rethrows -> R {
    try body(self.handle)
  }
}

// MARK: - XGRGrammar

/// An XGrammar grammar.
public struct XGRGrammar: ~Copyable, @unchecked Sendable {
  // ``@unchecked Sendable`` is safe because XGrammar grammar handles are immutable after
  // construction; this reference type owns and destroys its handle exactly once.

  /// A configuration for ``jsonSchema(_:configuration:)``.
  public struct JSONSchemaConfiguration: Hashable, Sendable {
    /// A type for the separators of a JSON schema grammar. (eg. `[":", ","]`)
    public struct Separators: Hashable, Sendable {
      /// The comma separator.
      public var comma: String

      /// The colon separator.
      public var colon: String

      /// Creates separators.
      ///
      /// - Parameters:
      ///   - comma: The comma separator.
      ///   - colon: The colon separator.
      public init(comma: String, colon: String) {
        self.comma = comma
        self.colon = colon
      }
    }

    /// Whether or not whitespace characters are allowed around structural JSON elements (eg. commas and colons).
    ///
    /// When true ``indent`` and ``separators`` are ignored.
    public var anyWhitespace: Bool

    /// The number of spaces for indentation.
    ///
    /// This value defaults to 2 if left nil.
    public var indent: Int?

    /// The ``Separators`` to use.
    public var separators: Separators?

    /// Whether strict mode is enabled for this schema.
    ///
    /// In strict mode, the generated grammar will not allow properties and items that is not
    /// specified in the schema.
    public var isStrict: Bool

    /// The maximum number of whitespace tokens.
    public var maximumWhitespaceCount: Int?

    /// Whether or not the object keys can be generated in any order.
    public var anyOrder: Bool

    /// Creates a JSON schema configuration.
    ///
    /// - Parameters:
    ///   - anyWhitespace: Whether or not whitespace characters are allowed around structural JSON
    ///     elements (eg. commas and colons).
    ///   - indent: The number of spaces for indentation.
    ///   - separators: The ``Separators`` to use.
    ///   - isStrict: Whether strict mode is enabled for this schema.
    ///   - maximumWhitespaceCount: The maximum number of whitespace tokens.
    ///   - anyOrder: Whether or not the object keys can be generated in any order.
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

  /// The underlying grammar pointer.
  public let handle: xgrammar_grammar_t

  /// Constructs a gramamr from EBNF syntax.
  ///
  /// The supported syntax EBNF follows [GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).
  ///
  /// - Parameters:
  ///   - ebnf: The EBNF syntax.
  ///   - rootRuleName: The name of the root rule.
  /// - Returns: An ``XGRGrammar``.
  public static func ebnf(_ ebnf: String, rootRuleName: String = "root") throws -> XGRGrammar {
    let handle = try ebnf.withCString { ebnf in
      try rootRuleName.withCString { try require(xgrammar_grammar_init_ebnf(ebnf, $0)) }
    }
    return Self(handle: handle)
  }

  /// Converts a regular expression into an equivalent grammar.
  ///
  /// - Parameters:
  ///   - regex: A regex string.
  /// - Returns: An ``XGRGrammar``.
  public static func regex(_ regex: String) throws -> XGRGrammar {
    let handle = try regex.withCString { try require(xgrammar_grammar_init_regex($0)) }
    return Self(handle: handle)
  }

  /// Parses a Lark grammar, optionally resolving tokenizer tokens and named grammar references.
  ///
  /// - Parameters:
  ///   - lark: The Lark syntax.
  ///   - tokenizerInfo: An optional ``XGRTokenizerInfo``.
  ///   - namedGrammars: Any ``XGRNamedGrammar`` instances the lark syntax references.
  /// - Returns: An ``XGRGrammar``.
  public static func lark(
    _ lark: String,
    tokenizerInfo: XGRTokenizerInfo? = nil,
    namedGrammars: [XGRNamedGrammar] = []
  ) throws -> XGRGrammar {
    let names = namedGrammars.map { $0.name }
    let sources = namedGrammars.map { namedGrammar in
      if case .lark(let source) = namedGrammar.grammar { source } else { "" }
    }
    let descriptors = namedGrammars.map { $0.rawValue }
    let handle = try withCopiedCStringPointerBuffer(names) { names in
      try withCopiedCStringPointerBuffer(sources) { sources in
        var descriptors = descriptors
        for index in descriptors.indices {
          descriptors[index].name = names[index]
          descriptors[index].lark_source = sources[index]
        }
        return try descriptors.withUnsafeBufferPointer { descriptors in
          try lark.withCString {
            try require(
              xgrammar_grammar_init_lark(
                $0,
                tokenizerInfo?.handle,
                descriptors.baseAddress,
                descriptors.count
              )
            )
          }
        }
      }
    }
    return Self(handle: handle)
  }

  /// Builds a grammar from a structural tag JSON specification.
  ///
  /// See [this](https://xgrammar.mlc.ai/docs/structural_tag/structural_tag.html) for more info on
  /// structural tag syntax.
  ///
  /// - Parameters:
  ///   - structuralTagJSON: The raw structural tag JSON.
  ///   - tokenizerInfo: Optional ``XGRTokenizerInfo``.
  /// - Returns: An ``XGRGrammar``.
  public static func structuralTagJSON(
    _ structuralTagJSON: String,
    tokenizerInfo: XGRTokenizerInfo? = nil
  ) throws -> XGRGrammar {
    let handle = try structuralTagJSON.withCString {
      try require(
        xgrammar_grammar_init_structural_tag($0, tokenizerInfo?.handle)
      )
    }
    return Self(handle: handle)
  }

  /// Builds a grammar from a raw JSON schema string.
  ///
  /// - Parameters:
  ///   - jsonSchema: The raw JSON schema string.
  ///   - configuration: The ``JSONSchemaConfiguration`` to use.
  /// - Returns: An ``XGRGrammar``.
  public static func jsonSchema(
    _ jsonSchema: String,
    configuration: JSONSchemaConfiguration = JSONSchemaConfiguration()
  ) throws -> XGRGrammar {
    guard configuration.indent.map({ $0 >= 0 }) ?? true,
      configuration.maximumWhitespaceCount.map({ $0 >= 0 }) ?? true
    else {
      throw XGRError.invalidJSONSchemaConfiguration
    }
    let indent = try configuration.indent.map {
      try Int32($0, error: .invalidJSONSchemaConfiguration)
    }
    let maximumWhitespaceCount = try configuration.maximumWhitespaceCount.map {
      try Int32($0, error: .invalidJSONSchemaConfiguration)
    }
    let handle = try jsonSchema.withCString { schema in
      if let separators = configuration.separators {
        return try separators.comma.withCString { comma in
          try separators.colon.withCString { colon in
            try require(
              xgrammar_grammar_init_json_schema(
                schema,
                configuration.anyWhitespace.intValue,
                indent ?? -1,
                comma,
                colon,
                configuration.isStrict.intValue,
                maximumWhitespaceCount ?? -1,
                configuration.anyOrder.intValue
              )
            )
          }
        }
      }
      return try require(
        xgrammar_grammar_init_json_schema(
          schema,
          configuration.anyWhitespace.intValue,
          indent ?? -1,
          nil,
          nil,
          configuration.isStrict.intValue,
          maximumWhitespaceCount ?? -1,
          configuration.anyOrder.intValue
        )
      )
    }
    return Self(handle: handle)
  }

  /// Creates a grammar from its serialized JSON representation.
  ///
  /// - Parameters:
  ///   - serializedJSON: The serialized JSON grammar string.
  /// - Returns: An ``XGRGrammar``.
  public static func serializedJSON(_ serializedJSON: String) throws -> XGRGrammar {
    let handle = try serializedJSON.withCString {
      try require(xgrammar_grammar_deserialize_json($0))
    }
    return Self(handle: handle)
  }

  /// Creates a grammar from a raw pointer.
  ///
  /// - Parameters:
  ///   - handle: The underlying grammar pointer.
  public init(handle: consuming xgrammar_grammar_t) {
    self.handle = consume handle
  }

  deinit { xgrammar_grammar_destroy(self.handle) }

  /// Creates a grammar that represents any JSON string.
  ///
  /// - Returns: An ``XGRGrammar``.
  public static func builtinJSONGrammar() -> XGRGrammar {
    XGRGrammar(handle: xgrammar_grammar_builtin_json()!)
  }

  /// The normalized EBNF representation of the grammar.
  public var ebnf: String {
    let capacity = xgrammar_grammar_ebnf(self.handle, nil, 0)
    return xgrammarString(
      { xgrammar_grammar_ebnf(self.handle, $0, $1) },
      capacity: capacity
    )
  }

  /// Serializes the grammar or compiled grammar to a JSON string.
  public func serializedJSON() throws -> String {
    try xgrammarString { xgrammar_grammar_serialize_json(self.handle, $0, $1) }
  }

  /// Creates an independent grammar with the same grammar representation.
  public func copy() throws -> XGRGrammar {
    XGRGrammar(handle: try require(xgrammar_grammar_copy(self.handle)))
  }

  /// Produces a concatenated grammar with this grammar and another grammar.
  ///
  /// - Parameters:
  ///   - grammar: The other grammar.
  /// - Returns: An ``XGRGrammar``.
  public func concatenate(_ grammar: borrowing XGRGrammar) throws -> XGRGrammar {
    try EdgeToolsXGrammar.concatenate(self, grammar)
  }

  /// Produces a unioned grammar with this grammar and another grammar.
  ///
  /// - Parameters:
  ///   - grammar: The other grammar.
  /// - Returns: An ``XGRGrammar``.
  public func union(_ grammar: borrowing XGRGrammar) throws -> XGRGrammar {
    try EdgeToolsXGrammar.union(self, grammar)
  }

  /// Concatenates two grammars.
  ///
  /// The resulting grammar accepts an input only when it first matches ``lhs`` and then matches
  /// ``rhs``.
  ///
  /// - Parameters:
  ///   - lhs: The grammar that matches first.
  ///   - rhs: The grammar that matches after ``lhs``.
  /// - Returns: A grammar representing the concatenation.
  public static func + (
    lhs: borrowing XGRGrammar,
    rhs: borrowing XGRGrammar
  ) throws -> XGRGrammar {
    try EdgeToolsXGrammar.concatenate(lhs, rhs)
  }

  /// Creates a grammar accepting either of two grammars.
  ///
  /// - Parameters:
  ///   - lhs: One accepted alternative.
  ///   - rhs: The other accepted alternative.
  /// - Returns: A grammar representing the union of both alternatives.
  public static func || (
    lhs: borrowing XGRGrammar,
    rhs: borrowing XGRGrammar
  ) throws -> XGRGrammar {
    try EdgeToolsXGrammar.union(lhs, rhs)
  }

  /// Optionalizes this grammar
  ///
  /// - Returns: An ``XGRGrammar``.
  public func optional() throws -> XGRGrammar {
    try XGRGrammar(handle: require(xgrammar_grammar_optional(self.handle)))
  }

  /// Repeats this grammar exactly `count` times.
  ///
  /// - Returns: An ``XGRGrammar``.
  public func repeated(exactly count: Int) throws -> XGRGrammar {
    try repeatGrammar(self, exactly: count)
  }

  /// Repeats this grammar based on a `ClosedRange`.
  ///
  /// - Parameters:
  ///   - range: The repetition range.
  /// - Returns: An ``XGRGrammar``.
  public func repeated(_ range: ClosedRange<Int>) throws -> XGRGrammar {
    try repeatGrammar(self, range)
  }

  /// Repeats this grammar based on a `PartialRangeFrom`.
  ///
  /// - Parameters:
  ///   - range: The repetition range.
  /// - Returns: An ``XGRGrammar``.
  public func repeated(_ range: PartialRangeFrom<Int>) throws -> XGRGrammar {
    try repeatGrammar(self, range)
  }
}

// MARK: - XGRNamedGrammar

/// A names grammar type for use in ``XGRGrammar/lark(_:tokenizerInfo:namedGrammars:)``.
public struct XGRNamedGrammar: RawRepresentable, Sendable {
  /// The actual grammar representation.
  public enum Grammar: Sendable {
    /// Uses an already-created grammar.
    case grammar(Reference)

    /// Uses a Lark grammar source.
    case lark(String)

    /// Owns a grammar for storage in a copyable named-grammar descriptor.
    public final class Reference: @unchecked Sendable {
      public let grammar: XGRGrammar

      public init(grammar: consuming XGRGrammar) {
        self.grammar = consume grammar
      }
    }

    /// Stores an owned grammar in a named-grammar descriptor.
    public static func grammar(_ grammar: consuming XGRGrammar) -> Self {
      .grammar(Reference(grammar: consume grammar))
    }
  }

  /// The name of the grammar.
  public let name: String

  /// The actual ``Grammar`` content.
  public let grammar: Grammar

  /// Creates a named grammar.
  ///
  /// - Parameters:
  ///   - name: The name of the grammar.
  ///   - grammar: The actual ``Grammar`` content.
  public init(name: String, grammar: Grammar) {
    self.name = name
    self.grammar = grammar
  }

  public var rawValue: xgrammar_named_grammar_t {
    switch self.grammar {
    case .grammar(let reference):
      xgrammar_named_grammar_t(
        name: nil,
        kind: xgrammar_named_grammar_handle,
        lark_source: nil,
        grammar: reference.grammar.handle
      )
    case .lark:
      xgrammar_named_grammar_t(
        name: nil,
        kind: xgrammar_named_grammar_lark,
        lark_source: nil,
        grammar: nil
      )
    }
  }

  public init?(rawValue: xgrammar_named_grammar_t) {
    guard let name = rawValue.name else { return nil }

    let grammar: Grammar
    switch rawValue.kind {
    case xgrammar_named_grammar_handle:
      guard
        let handle = rawValue.grammar,
        let serializedJSON = try? xgrammarString({
          xgrammar_grammar_serialize_json(handle, $0, $1)
        }),
        let grammarValue = try? XGRGrammar.serializedJSON(serializedJSON)
      else { return nil }
      grammar = .grammar(grammarValue)
    case xgrammar_named_grammar_lark:
      guard let source = rawValue.lark_source else { return nil }
      grammar = .lark(String(cString: source))
    default: return nil
    }

    self.init(name: String(cString: name), grammar: grammar)
  }
}

// MARK: - XGRCompiledGrammar

/// A grammar preprocessed for a particular tokenizer vocabulary.
///
/// Use ``XGRCompiler/compile(_:)`` to create a compiled grammar before constructing an
/// ``XGRMatcher``. Compilation precomputes the token-level information required for efficient
/// constrained generation.
public struct XGRCompiledGrammar: ~Copyable, @unchecked Sendable {
  /// The underlying compiled-grammar pointer.
  public let handle: xgrammar_compiled_grammar_t

  /// Creates a compiled grammar from a raw pointer.
  ///
  /// - Parameters:
  ///   - handle: The underlying compiled-grammar pointer whose ownership is transferred.
  public init(handle: consuming xgrammar_compiled_grammar_t) {
    self.handle = consume handle
  }

  /// Restores a compiled grammar from its serialized JSON representation.
  ///
  /// The tokenizer info must describe the same vocabulary used when the grammar was serialized.
  ///
  /// - Parameters:
  ///   - serializedJSON: The serialized compiled-grammar JSON string.
  ///   - tokenizerInfo: The tokenizer metadata associated with the compiled grammar.
  public init(serializedJSON: String, tokenizerInfo: XGRTokenizerInfo) throws {
    self.handle = try serializedJSON.withCString {
      try require(xgrammar_compiled_grammar_deserialize_json($0, tokenizerInfo.handle))
    }
  }

  deinit { xgrammar_compiled_grammar_destroy(self.handle) }

  /// The grammar used to produce this compiled grammar.
  public var grammar: XGRGrammar {
    XGRGrammar(handle: xgrammar_compiled_grammar_grammar(self.handle)!)
  }

  /// The tokenizer metadata used during compilation.
  public var tokenizerInfo: XGRTokenizerInfo {
    XGRTokenizerInfo(handle: xgrammar_compiled_grammar_tokenizer_info(self.handle)!)
  }

  /// An approximate measure of the compiled grammar memory usage, in bytes.
  public var memorySizeBytes: Int64 {
    xgrammar_compiled_grammar_memory_size_bytes(self.handle)
  }

  /// Serializes this compiled grammar to JSON for persistence or transport.
  ///
  /// - Returns: The serialized compiled-grammar JSON string.
  public func serializedJSON() throws -> String {
    try xgrammarString { xgrammar_compiled_grammar_serialize_json(self.handle, $0, $1) }
  }

  /// Creates an independent compiled grammar with the same grammar representation.
  public func copy() throws -> XGRCompiledGrammar {
    XGRCompiledGrammar(handle: try require(xgrammar_compiled_grammar_copy(self.handle)))
  }
}

// MARK: - XGRCompiler

/// Compiles grammars for a single tokenizer vocabulary and caches the results.
///
/// Create one compiler per vocabulary. Reusing a compiler avoids repeating preprocessing when the
/// same grammar or schema is compiled more than once.
public struct XGRCompiler: ~Copyable, @unchecked Sendable {
  /// The underlying compiler pointer.
  public let handle: xgrammar_compiler_t

  /// The tokenizer metadata used for every compilation by this compiler.
  public let tokenizerInfo: XGRTokenizerInfo

  /// Creates a compiler for a tokenizer vocabulary.
  ///
  /// - Parameters:
  ///   - tokenizerInfo: The tokenizer metadata used for compiled grammars.
  ///   - maxThreads: The maximum number of threads used during compilation. Must be positive.
  ///   - cacheEnabled: Whether compiled grammars are cached for reuse.
  ///   - maxMemoryBytes: The cache memory limit in bytes. A negative value leaves the limit unset.
  public init(
    tokenizerInfo: XGRTokenizerInfo,
    maxThreads: Int = 8,
    cacheEnabled: Bool = true,
    maxMemoryBytes: Int64 = -1
  ) throws {
    let maxThreads = try Int32(maxThreads, error: .invalidCompilerConfiguration)
    guard maxThreads > 0 else { throw XGRError.invalidCompilerConfiguration }
    let handle = xgrammar_compiler_init(
      tokenizerInfo.handle,
      maxThreads,
      cacheEnabled.intValue,
      maxMemoryBytes
    )
    self.handle = try require(handle)
    self.tokenizerInfo = tokenizerInfo
  }

  /// Creates a compiler from a raw pointer and its tokenizer metadata.
  ///
  /// - Parameters:
  ///   - handle: The underlying compiler pointer whose ownership is transferred.
  ///   - tokenizerInfo: The tokenizer metadata used by the compiler.
  public init(handle: consuming xgrammar_compiler_t, tokenizerInfo: XGRTokenizerInfo) {
    self.handle = consume handle
    self.tokenizerInfo = tokenizerInfo
  }

  deinit { xgrammar_compiler_destroy(self.handle) }

  /// Creates a copy of this compiler and its tokenizer info.
  public func copy() throws -> XGRCompiler {
    XGRCompiler(
      handle: try require(xgrammar_compiler_copy(self.handle)),
      tokenizerInfo: try self.tokenizerInfo.copy()
    )
  }

  /// The current approximate cache size, in bytes.
  public var cacheSizeBytes: Int64 {
    xgrammar_compiler_cache_size_bytes(self.handle)
  }

  /// The configured maximum cache size, in bytes.
  public var cacheLimitBytes: Int64 {
    xgrammar_compiler_cache_limit_bytes(self.handle)
  }

  /// Removes all compiled grammars from the compiler cache.
  public func clearCache() {
    xgrammar_compiler_clear_cache(self.handle)
  }

  /// Compiles a grammar for this compiler’s tokenizer vocabulary.
  ///
  /// The compiler may return a cached result when it has already compiled an equivalent grammar.
  ///
  /// - Parameters:
  ///   - grammar: The grammar to preprocess.
  /// - Returns: A compiled grammar ready to create an ``XGRMatcher``.
  public func compile(_ grammar: borrowing XGRGrammar) throws -> XGRCompiledGrammar {
    let handle = try require(xgrammar_compiler_compile_grammar(self.handle, grammar.handle))
    return XGRCompiledGrammar(handle: handle)
  }
}

// MARK: - XGRMatcher

/// Tracks incremental constrained generation against a compiled grammar.
///
/// Query ``bitmask()`` to determine which tokens are valid next, then call ``accept(tokenId:)``
/// after selecting a token. A matcher can be reset, rolled back, or forked to manage generation
/// branches.
public struct XGRMatcher: ~Copyable, @unchecked Sendable {
  /// The underlying matcher pointer.
  public let handle: xgrammar_matcher_t

  /// Creates a matcher for a compiled grammar.
  ///
  /// - Parameters:
  ///   - compiledGrammar: The grammar and tokenizer vocabulary to constrain generation against.
  ///   - overrideStopTokenIDs: Token IDs that override the tokenizer’s configured stop tokens.
  ///     An empty array uses the stop tokens from the compiled grammar’s tokenizer info.
  ///   - terminateWithoutStopToken: Whether the matcher may terminate after a complete match
  ///     without requiring a stop token.
  ///   - maxRollbackTokens: The maximum number of accepted tokens retained for rollback. Use a
  ///     negative value for XGrammar’s unlimited rollback behavior.
  public init(
    compiledGrammar: borrowing XGRCompiledGrammar,
    overrideStopTokenIDs: [Int] = [],
    terminateWithoutStopToken: Bool = false,
    maxRollbackTokens: Int = -1
  ) throws {
    let maxRollbackTokens = try Int32(maxRollbackTokens, error: .invalidMatcherConfiguration)
    let overrideStopTokenIDs = try overrideStopTokenIDs.map {
      try Int32($0, error: .invalidMatcherConfiguration)
    }

    self.handle = try overrideStopTokenIDs.withUnsafeBufferPointer {
      try require(
        xgrammar_matcher_init(
          compiledGrammar.handle,
          $0.baseAddress,
          $0.count,
          terminateWithoutStopToken.intValue,
          maxRollbackTokens
        )
      )
    }
  }

  /// Creates a matcher from a raw pointer.
  ///
  /// - Parameters:
  ///   - handle: The underlying matcher pointer whose ownership is transferred.
  public init(handle: consuming xgrammar_matcher_t) {
    self.handle = consume handle
  }

  deinit { xgrammar_matcher_destroy(self.handle) }

  /// Indicates whether the matcher has accepted a complete valid sequence.
  public var isCompleted: Bool {
    xgrammar_matcher_is_completed(self.handle) != 0
  }

  /// Indicates whether generation has been terminated by the matcher.
  public var isTerminated: Bool {
    xgrammar_matcher_is_terminated(self.handle) != 0
  }

  /// Returns the vocabulary acceptance mask for the current matcher state.
  ///
  /// Each bit corresponds to a token ID: a set bit indicates that accepting that token preserves a
  /// valid continuation. Bits are packed into `Int32` words in token-ID order.
  ///
  /// - Returns: A bitmask of token IDs accepted at the current state, or `nil` when every token is
  ///   accepted.
  public func bitmask() -> [Int32]? {
    let bitCount = Int(xgrammar_matcher_bit_count(self.handle))
    var bitmask = [Int32](repeating: 0, count: (bitCount + 31) / 32)
    let isRequired = bitmask.withUnsafeMutableBufferPointer { bitmask in
      xgrammar_matcher_bitmask(self.handle, bitmask.baseAddress)
    } != 0
    return isRequired ? bitmask : nil
  }

  /// Advances the matcher with a token ID.
  ///
  /// - Parameters:
  ///   - tokenId: The tokenizer token ID to accept.
  /// - Returns: Whether the token is accepted by the grammar at the current state.
  @discardableResult
  public func accept(tokenId: Int) -> Bool {
    xgrammar_matcher_accept_token(self.handle, Int32(tokenId)) != 0
  }

  /// Advances the matcher with a string.
  ///
  /// This is useful when matching text that is not already tokenized.
  ///
  /// - Parameters:
  ///   - string: The string to accept.
  /// - Returns: Whether the string is accepted by the grammar at the current state.
  @discardableResult
  public func accept(string: String) -> Bool {
    string.withCString { string in
      xgrammar_matcher_accept_string(self.handle, string) != 0
    }
  }

  /// Rewinds the matcher by the specified number of accepted tokens.
  ///
  /// Rollback is limited by ``init(compiledGrammar:overrideStopTokenIDs:terminateWithoutStopToken:maxRollbackTokens:)``.
  ///
  /// - Parameters:
  ///   - tokenCount: The number of most recently accepted tokens to remove.
  public func rollback(_ tokenCount: Int = 1) {
    xgrammar_matcher_rollback(self.handle, Int32(tokenCount))
  }

  /// Restores the matcher to its initial state.
  public func reset() {
    xgrammar_matcher_reset(self.handle)
  }

  /// Creates an independent matcher with the current state.
  ///
  /// Subsequent changes to either matcher do not affect the other.
  ///
  /// - Returns: A matcher initialized with this matcher’s current state.
  public func fork() -> XGRMatcher {
    XGRMatcher(handle: xgrammar_matcher_fork(self.handle)!)
  }
}

// MARK: - Grammar Combinators

/// Concatenates two grammars.
///
/// The resulting grammar accepts an input only when it first matches ``lhs`` and then matches
/// ``rhs``.
///
/// - Parameters:
///   - lhs: The grammar that matches first.
///   - rhs: The grammar that matches after ``lhs``.
/// - Returns: A grammar representing the concatenation.
public func concatenate(
  _ lhs: borrowing XGRGrammar,
  _ rhs: borrowing XGRGrammar
) throws -> XGRGrammar {
  try concatenatedGrammar(handles: [lhs.handle, rhs.handle])
}

/// Creates a grammar accepting either of two grammars.
///
/// - Parameters:
///   - lhs: One accepted alternative.
///   - rhs: The other accepted alternative.
/// - Returns: A grammar representing the union of both alternatives.
public func union(
  _ lhs: borrowing XGRGrammar,
  _ rhs: borrowing XGRGrammar
) throws -> XGRGrammar {
  try unionedGrammar(handles: [lhs.handle, rhs.handle])
}

private func concatenatedGrammar(handles: [xgrammar_grammar_t?]) throws -> XGRGrammar {
  let handle = try handles.withUnsafeBufferPointer {
    try require(xgrammar_grammar_concatenate($0.baseAddress, $0.count))
  }
  return XGRGrammar(handle: handle)
}

private func unionedGrammar(handles: [xgrammar_grammar_t?]) throws -> XGRGrammar {
  let handle = try handles.withUnsafeBufferPointer {
    try require(xgrammar_grammar_union($0.baseAddress, $0.count))
  }
  return XGRGrammar(handle: handle)
}

/// Repeats a grammar exactly a specified number of times.
///
/// A count of zero creates a grammar that accepts the empty string.
///
/// - Parameters:
///   - grammar: The grammar to repeat.
///   - count: The exact number of repetitions. It must not be negative.
/// - Returns: A grammar representing the repeated input.
public func repeatGrammar(_ grammar: borrowing XGRGrammar, exactly count: Int) throws -> XGRGrammar {
  try repeatGrammar(grammar, count...count)
}

/// Repeats a grammar within a closed range.
///
/// The lower bound must be nonnegative. A lower bound of zero permits the empty string.
///
/// - Parameters:
///   - grammar: The grammar to repeat.
///   - range: The inclusive range of permitted repetition counts.
/// - Returns: A grammar representing the bounded repetition.
public func repeatGrammar(_ grammar: borrowing XGRGrammar, _ range: ClosedRange<Int>) throws -> XGRGrammar {
  guard range.lowerBound >= 0 else { throw XGRError.invalidRepetitionRange }
  let lowerBound = try Int32(range.lowerBound, error: .invalidRepetitionRange)
  let upperBound = try Int32(range.upperBound, error: .invalidRepetitionRange)
  let handle = try require(xgrammar_grammar_repeat(grammar.handle, lowerBound, upperBound))
  return XGRGrammar(handle: handle)
}

/// Repeats a grammar at least a specified number of times.
///
/// The lower bound must be nonnegative. A lower bound of zero permits the empty string.
///
/// - Parameters:
///   - grammar: The grammar to repeat.
///   - range: The lower bound of the unbounded repetition range.
/// - Returns: A grammar representing the unbounded repetition.
public func repeatGrammar(
  _ grammar: borrowing XGRGrammar,
  _ range: PartialRangeFrom<Int>
) throws -> XGRGrammar {
  guard range.lowerBound >= 0 else { throw XGRError.invalidRepetitionRange }
  let lowerBound = try Int32(range.lowerBound, error: .invalidRepetitionRange)
  let handle = try require(xgrammar_grammar_repeat(grammar.handle, lowerBound, -1))
  return XGRGrammar(handle: handle)
}

// MARK: - Helpers

private func withCopiedCStringPointerBuffer<Result>(
  _ strings: [String],
  _ body: (UnsafeMutableBufferPointer<UnsafePointer<CChar>?>) throws -> Result
) rethrows -> Result {
  var mutablePointers = strings.map {
    let destination = UnsafeMutablePointer<CChar>.allocate(capacity: $0.utf8CString.count)
    for index in 0..<$0.utf8CString.count {
      destination[index] = $0.utf8CString[index]
    }
    return UnsafePointer<CChar>(destination) as UnsafePointer<CChar>?
  }
  defer {
    for pointer in mutablePointers {
      pointer?.deallocate()
    }
  }
  return try mutablePointers.withUnsafeMutableBufferPointer { try body($0) }
}

extension Bool {
  fileprivate var intValue: Int32 {
    self ? 1 : 0
  }
}

private func require<Handle>(_ handle: Handle?) throws -> Handle {
  guard let handle else {
    throw XGRError(
      code: .xgrammarFailure,
      message: String(cString: xgrammar_last_error_message())
    )
  }
  return handle
}

extension Int32 {
  fileprivate init(_ value: Int, error: XGRError) throws {
    guard let value = Int32(exactly: value) else { throw error }
    self = value
  }
}

private func xgrammarString(_ body: (UnsafeMutablePointer<CChar>?, Int) -> Int) throws -> String {
  let capacity = body(nil, 0)
  guard capacity > 0 else {
    throw XGRError(
      code: .xgrammarFailure,
      message: String(cString: xgrammar_last_error_message())
    )
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
