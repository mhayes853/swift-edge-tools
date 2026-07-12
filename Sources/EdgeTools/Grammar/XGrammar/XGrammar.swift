#if XGrammar
  import CXGrammar
  import Foundation

  // MARK: - XGrammarVocabularyType

  public enum XGrammarVocabularyType: Hashable, Sendable {
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

  // MARK: - XGrammarError

  public struct XGrammarError: Error, Hashable, Sendable {
    public let message: String

    public init(message: String) {
      self.message = message
    }

    public static let invalidCompilerConfiguration = Self(
      message: "Invalid XGrammar compiler configuration."
    )
    public static let invalidRepetitionRange = Self(message: "Invalid XGrammar repetition range.")
    public static let emptyGrammarCollection = Self(message: "Expected at least one grammar.")
  }

  // MARK: - XGrammarCompiler

  public struct XGrammarCompiler: ~Copyable {
    public struct Configuration: Hashable, Sendable {
      public var memoryLimitBytes: Int64?
      public var maximumThreads: Int?
      public var isCacheEnabled: Bool

      public init(
        memoryLimitBytes: Int64? = nil,
        maximumThreads: Int? = nil,
        isCacheEnabled: Bool = true
      ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.maximumThreads = maximumThreads
        self.isCacheEnabled = isCacheEnabled
      }
    }

    private let handle: xgrammar_compiler_t
    private let bitmaskWordCount: Int

    public var cacheSizeBytes: Int64 {
      xgrammar_compiler_cache_size_bytes(self.handle)
    }

    public var cacheLimitBytes: Int64 {
      xgrammar_compiler_cache_limit_bytes(self.handle)
    }

    public init(
      encodedVocabulary: [String],
      vocabularyType: XGrammarVocabularyType = .raw,
      vocabularySize: Int? = nil,
      stopTokenIDs: [Int] = [],
      addPrefixSpace: Bool = false,
      configuration: Configuration = Configuration()
    ) throws {
      try Self.validate(configuration: configuration)
      guard vocabularySize.map({ $0 >= 0 }) ?? true,
        stopTokenIDs.allSatisfy({ Int32(exactly: $0) != nil })
      else {
        throw XGrammarError.invalidCompilerConfiguration
      }

      let resolvedVocabularySize = vocabularySize ?? encodedVocabulary.count
      let handle = try withCopiedCStringPointerBuffer(encodedVocabulary) { vocabulary in
        try stopTokenIDs.map(Int32.init)
          .withUnsafeBufferPointer { stopTokenIDs in
            let compiler = xgrammar_compiler_init(
              vocabulary.baseAddress,
              vocabulary.count,
              vocabularyType.rawValue,
              Int32(vocabularySize ?? -1),
              stopTokenIDs.baseAddress,
              stopTokenIDs.count,
              addPrefixSpace.intValue(as: Int32.self)
            )
            return try Self.requiredHandle(compiler)
          }
      }
      Self.apply(configuration, to: handle)
      self.handle = handle
      self.bitmaskWordCount = (resolvedVocabularySize + 31) / 32
    }

    deinit {
      xgrammar_compiler_destroy(self.handle)
    }

    public func apply(configuration: Configuration) throws {
      try Self.validate(configuration: configuration)
      Self.apply(configuration, to: self.handle)
    }

    private static func apply(
      _ configuration: Configuration,
      to handle: xgrammar_compiler_t
    ) {
      xgrammar_compiler_set_memory_limit(handle, configuration.memoryLimitBytes ?? -1)
      xgrammar_compiler_set_max_threads(handle, Int64(configuration.maximumThreads ?? -1))
      xgrammar_compiler_set_cache_enabled(
        handle,
        configuration.isCacheEnabled.intValue(as: Int32.self)
      )
    }

    public func clearCache() {
      xgrammar_compiler_clear_cache(self.handle)
    }

    public borrowing func compile(
      _ grammar: borrowing XGrammarGrammar
    ) throws -> XGrammarMatcher {
      try XGrammarMatcher(
        handle: Self.requiredHandle(xgrammar_compile_matcher(self.handle, grammar.handle)),
        bitmaskWordCount: self.bitmaskWordCount
      )
    }

    private static func validate(configuration: Configuration) throws {
      guard configuration.memoryLimitBytes.map({ $0 >= 0 }) ?? true,
        configuration.maximumThreads.map({ $0 > 0 }) ?? true
      else {
        throw XGrammarError.invalidCompilerConfiguration
      }
    }

    static func requiredHandle<Handle>(_ handle: Handle?) throws -> Handle {
      guard let handle else {
        throw XGrammarError(message: String(cString: xgrammar_last_error_message()))
      }
      return handle
    }
  }

  // MARK: - XGrammarGrammar

  public struct XGrammarGrammar: ~Copyable {
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

    let handle: xgrammar_grammar_t

    public init(ebnf: String, rootRuleName: String = "root") throws {
      self.handle = try ebnf.withCString { ebnf in
        try rootRuleName.withCString { rootRuleName in
          try XGrammarCompiler.requiredHandle(xgrammar_grammar_init_ebnf(ebnf, rootRuleName))
        }
      }
    }

    public init(regex: String) throws {
      self.handle = try regex.withCString {
        try XGrammarCompiler.requiredHandle(xgrammar_grammar_init_regex($0))
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
      else {
        throw XGrammarError.invalidCompilerConfiguration
      }
      self.handle = try jsonSchema.withCString { jsonSchema in
        if let separators = configuration.separators {
          return try separators.comma.withCString { comma in
            try separators.colon.withCString { colon in
              try XGrammarCompiler.requiredHandle(
                xgrammar_grammar_init_json_schema(
                  jsonSchema,
                  configuration.anyWhitespace.intValue(as: Int32.self),
                  Int32(configuration.indent ?? -1),
                  comma,
                  colon,
                  configuration.isStrict.intValue(as: Int32.self),
                  Int32(configuration.maximumWhitespaceCount ?? -1),
                  configuration.anyOrder.intValue(as: Int32.self)
                )
              )
            }
          }
        }
        return try XGrammarCompiler.requiredHandle(
          xgrammar_grammar_init_json_schema(
            jsonSchema,
            configuration.anyWhitespace.intValue(as: Int32.self),
            Int32(configuration.indent ?? -1),
            nil,
            nil,
            configuration.isStrict.intValue(as: Int32.self),
            Int32(configuration.maximumWhitespaceCount ?? -1),
            configuration.anyOrder.intValue(as: Int32.self)
          )
        )
      }
    }

    init(handle: xgrammar_grammar_t) {
      self.handle = handle
    }

    deinit {
      xgrammar_grammar_destroy(self.handle)
    }

    public var ebnf: String {
      get throws {
        let requiredCapacity = xgrammar_grammar_ebnf(self.handle, nil, 0)
        guard requiredCapacity > 0 else {
          throw XGrammarError(message: String(cString: xgrammar_last_error_message()))
        }
        var buffer = [CChar](repeating: 0, count: Int(requiredCapacity))
        let returnedCapacity = buffer.withUnsafeMutableBufferPointer {
          xgrammar_grammar_ebnf(self.handle, $0.baseAddress, $0.count)
        }
        guard returnedCapacity == requiredCapacity else {
          throw XGrammarError(message: String(cString: xgrammar_last_error_message()))
        }
        guard
          let ebnf = String(
            bytes: buffer.dropLast().map { UInt8(bitPattern: $0) },
            encoding: .utf8
          )
        else {
          throw XGrammarError(message: "XGrammar returned invalid UTF-8 EBNF.")
        }
        return ebnf
      }
    }

    private static func escapeEBNFLiteral(_ literal: String) -> String {
      literal.reduce(into: "") { result, character in
        if character == "\\" || character == "\"" {
          result.append("\\")
        }
        result.append(character)
      }
    }

    public borrowing func concatenate(
      _ grammar: borrowing XGrammarGrammar
    ) throws -> XGrammarGrammar {
      try EdgeTools.concatenate(self, grammar)
    }

    public borrowing func union(
      _ grammar: borrowing XGrammarGrammar
    ) throws -> XGrammarGrammar {
      try EdgeTools.union(self, grammar)
    }

    public borrowing func optional() throws -> XGrammarGrammar {
      try XGrammarGrammar(
        handle: XGrammarCompiler.requiredHandle(xgrammar_grammar_optional(self.handle))
      )
    }

    public borrowing func repeated(exactly count: Int) throws -> XGrammarGrammar {
      try `repeat`(self, exactly: count)
    }

    public borrowing func repeated(_ range: ClosedRange<Int>) throws -> XGrammarGrammar {
      try `repeat`(self, range)
    }

    public borrowing func repeated(_ range: PartialRangeFrom<Int>) throws -> XGrammarGrammar {
      try `repeat`(self, range)
    }
  }

  // MARK: - XGrammarMatcher

  public struct XGrammarMatcher: ~Copyable {
    private let handle: xgrammar_matcher_t
    private let bitmaskWordCount: Int

    public var isCompleted: Bool {
      xgrammar_matcher_is_completed(self.handle).boolValue
    }

    public var isTerminated: Bool {
      xgrammar_matcher_is_terminated(self.handle).boolValue
    }

    public var memorySizeBytes: Int64 {
      xgrammar_matcher_memory_size_bytes(self.handle)
    }

    fileprivate init(handle: xgrammar_matcher_t, bitmaskWordCount: Int) {
      self.handle = handle
      self.bitmaskWordCount = bitmaskWordCount
    }

    deinit {
      xgrammar_matcher_destroy(self.handle)
    }

    public func bitmask() -> GrammarBitmask {
      var bitmask = GrammarBitmask(bitCount: self.bitmaskWordCount * 32)
      _ = bitmask.storage.withUnsafeMutableBytes { bytes in
        let words = bytes.bindMemory(to: Int32.self)
        return xgrammar_matcher_bitmask(self.handle, words.baseAddress)
      }
      return bitmask
    }

    @discardableResult
    public func accept(tokenId: Int) -> Bool {
      xgrammar_matcher_accept_token(self.handle, Int32(tokenId)).boolValue
    }

    public func rollback(_ tokenCount: Int) {
      xgrammar_matcher_rollback(self.handle, Int32(tokenCount))
    }

    public func reset() {
      xgrammar_matcher_reset(self.handle)
    }

    public borrowing func fork() throws -> XGrammarMatcher {
      try XGrammarMatcher(
        handle: XGrammarCompiler.requiredHandle(xgrammar_matcher_fork(self.handle)),
        bitmaskWordCount: self.bitmaskWordCount
      )
    }
  }
#endif
