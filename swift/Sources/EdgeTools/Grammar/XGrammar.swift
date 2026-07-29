#if XGrammar
  import EdgeToolsXGrammar

  /// EdgeTools-specific XGrammar error codes.
  public struct EdgeToolsXGRError {
    /// An invalid ``GrammarToolCallRange`` for generation.
    public static let invalidToolInvocationRange = XGRError.Code(
      rawValue: "invalid-tool-invocation-range"
    )

    /// An empty tool collection for generation.
    public static let emptyToolCollection = XGRError.Code(rawValue: "empty-tool-collection")

    /// An unsupported tool schema for generation.
    public static let unsupportedToolSchema = XGRError.Code(rawValue: "unsupported-tool-schema")

    /// An incompatible tokenizer vocabulary for generation.
    public static let incompatibleTokenizerVocabulary = XGRError.Code(
      rawValue: "incompatible-tokenizer-vocabulary"
    )

    /// An invalid Needle tokenizer for generation.
    public static let invalidNeedleTokenizer = XGRError.Code(rawValue: "invalid-needle-tokenizer")
  }

  extension XGRError {
    static let invalidHuggingFaceMetadata = Self(
      code: .invalidHuggingFaceMetadata,
      message: "Invalid Hugging Face tokenizer metadata."
    )
    static let invalidToolInvocationRange = Self(
      code: EdgeToolsXGRError.invalidToolInvocationRange,
      message: "Tool invocation ranges cannot have a negative lower bound."
    )
    static let unsupportedToolSchema = Self(
      code: EdgeToolsXGRError.unsupportedToolSchema,
      message: "The tool definition has an unsupported schema."
    )
  }

  // MARK: - Tokenizer Info

  extension XGRTokenizerInfo {
    public static func huggingFace(
      encodedVocabulary: [String],
      backendJSON: String,
      modelVocabularySize: Int? = nil,
      stopTokenIDs: [Int] = []
    ) throws -> XGRTokenizerInfo {
      let metadata = try Self.metadata(huggingFaceBackendJSON: backendJSON)
      guard
        case .object(let decodedMetadata) = try EdgeToolsJSONDecoder()
          .decode(
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
  }

  // MARK: - Matcher

  extension XGRMatcher {
    public func grammarBitmask() -> GrammarBitmask {
      let words = self.bitmask()
      let storage = words.withUnsafeBytes { Array($0) }
      return GrammarBitmask(storage: storage)
    }
  }

  // MARK: - Grammar

  extension XGRGrammar {
    public static func literal(_ literal: String) throws -> XGRGrammar {
      let escapedLiteral = literal.reduce(into: "") { result, character in
        if character == "\\" || character == "\"" { result.append("\\") }
        result.append(character)
      }
      return try Self.ebnf("root ::= \"\(escapedLiteral)\"")
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
      guard
        let grammar = try? Self.structuralTagJSON(
          #"{"type":"structural_tag","format":{"type":"any_text"}}"#
        )
      else {
        preconditionFailure("XGrammar must support the any_text structural tag.")
      }
      return grammar
    }
  }

  // MARK: - EdgeToolsXGRGenerationConstraint

  /// Selects the grammar used to constrain an engine generation.
  @nonexhaustive
  public enum EdgeToolsXGRGenerationConstraint: Sendable {
    /// Allows arbitrary text output.
    case unconstrained

    /// Uses the supplied grammar directly.
    case grammar(XGRGrammar)

    /// Uses the engine's model-specific tool-call grammar.
    case toolsWithGrammar(
      range: GrammarToolCallRange = .unbounded(minimum: 0),
      grammar: (@Sendable (XGRGrammar, XGRTokenizerInfo) throws -> XGRGrammar)? = nil
    )

    /// The default tool-call constraint.
    public static let tools = Self.toolsWithGrammar()

    /// Constrains output to a value described by an EdgeTools generation schema.
    public static func schema(_ type: (some EdgeToolsGenerable).Type) -> Self {
      .grammar(.schema(type))
    }
  }

  // MARK: - EBNF

  struct XGREBNFDocument: Hashable, Sendable {
    struct Rule: Hashable, Sendable {
      let name: String
      var body: String
    }

    var rules: [Rule]

    init(_ source: String) throws {
      var rules = [Rule]()
      for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
        if let separator = line.firstRange(of: "::=") {
          let name = line[..<separator.lowerBound].trimmingWhitespace
          let body = line[separator.upperBound...].trimmingWhitespace
          guard !name.isEmpty else { throw XGRError.unsupportedToolSchema }
          rules.append(Rule(name: name, body: body))
        } else if !line.trimmingWhitespace.isEmpty {
          guard !rules.isEmpty else { throw XGRError.unsupportedToolSchema }
          rules[rules.count - 1].body += "\n" + line
        }
      }
      guard rules.contains(where: { $0.name == "root" }) else {
        throw XGRError.unsupportedToolSchema
      }
      self.rules = rules
    }

    var source: String {
      self.rules.map { "\($0.name) ::= \($0.body)" }.joined(separator: "\n")
    }

    mutating func removeDuplicateRules() {
      while true {
        let orderedRules = self.rules.sorted { lhs, _ in lhs.name == "root" }
        var canonicalNames = [String: String]()
        var aliases = [String: String]()
        var uniqueRules = [Rule]()

        for rule in orderedRules {
          if let canonicalName = canonicalNames[rule.body] {
            aliases[rule.name] = canonicalName
          } else {
            canonicalNames[rule.body] = rule.name
            uniqueRules.append(rule)
          }
        }
        guard !aliases.isEmpty else { return }
        for index in uniqueRules.indices {
          uniqueRules[index].body = Self.replacingRuleReferences(
            in: uniqueRules[index].body,
            aliases: aliases
          )
        }
        self.rules = uniqueRules
      }
    }

    mutating func mapLiterals(
      _ transform: (_ ruleName: String, _ value: String, _ suffix: Substring) -> String
    ) throws {
      for index in self.rules.indices {
        let rule = self.rules[index]
        self.rules[index].body = try Self.mapLiterals(in: rule.body) { value, suffix in
          transform(rule.name, value, suffix)
        }
      }
    }

    private static func mapLiterals(
      in source: String,
      _ transform: (_ value: String, _ suffix: Substring) -> String
    ) throws -> String {
      var output = ""
      var outputStart = source.startIndex

      for match in source.matches(of: literalRegex) {
        let literalRange = match.output.1.startIndex..<match.output.1.endIndex
        output.append(contentsOf: source[outputStart..<literalRange.lowerBound])
        let encodedValue = source[literalRange].dropFirst().dropLast()
        let value = Self.decodeLiteral(encodedValue)
        let transformed = transform(value, source[literalRange.upperBound...])
        output.append("\"")
        output.append(contentsOf: Self.escapeLiteral(transformed))
        output.append("\"")
        outputStart = literalRange.upperBound
      }
      output.append(contentsOf: source[outputStart...])
      return output
    }

    private static func replacingRuleReferences(
      in source: String,
      aliases: [String: String]
    ) -> String {
      let literalRanges = source.matches(of: literalRegex)
        .map { $0.output.1.startIndex..<$0.output.1.endIndex }
      var output = ""
      var outputStart = source.startIndex
      for match in source.matches(of: ruleReferenceRegex) {
        let range = match.range
        let isInsideLiteral = literalRanges.contains {
          $0.lowerBound <= range.lowerBound && range.upperBound <= $0.upperBound
        }
        guard !isInsideLiteral, let replacement = aliases[String(source[range])] else { continue }
        output.append(contentsOf: source[outputStart..<range.lowerBound])
        output.append(replacement)
        outputStart = range.upperBound
      }
      output.append(contentsOf: source[outputStart...])
      return output
    }

    private static func decodeLiteral(_ literal: Substring) -> String {
      var result = ""
      var isEscaping = false
      for character in literal {
        if isEscaping {
          switch character {
          case "n": result.append("\n")
          case "r": result.append("\r")
          case "t": result.append("\t")
          default: result.append(character)
          }
          isEscaping = false
        } else if character == "\\" {
          isEscaping = true
        } else {
          result.append(character)
        }
      }
      return result
    }

    private static func escapeLiteral(_ literal: String) -> String {
      literal.reduce(into: "") { result, character in
        switch character {
        case "\\", "\"":
          result.append("\\")
          result.append(character)
        case "\n": result.append(contentsOf: "\\n")
        case "\r": result.append(contentsOf: "\\r")
        case "\t": result.append(contentsOf: "\\t")
        default: result.append(character)
        }
      }
    }
  }

  extension Substring {
    var hasToolCallContinuationReference: Bool {
      self.contains(toolCallContinuationReferenceRegex)
    }
  }

  nonisolated(unsafe) private let literalRegex = /(?:^|[^\\])(\"(?:\\.|[^\"\\])*\")/

  nonisolated(unsafe) private let toolCallContinuationReferenceRegex =
    /\b(?:root(?:_[A-Za-z0-9]+)*|xml_object(?:_[A-Za-z0-9]+)*)\b/

  nonisolated(unsafe) private let ruleReferenceRegex = /\b[A-Za-z_][A-Za-z0-9_]*\b/

  // MARK: - Tool Call Building

  extension XGRGrammar {
    static func strictJSONArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      guard
        let grammar = try? XGRGrammar.jsonSchema(
          tool.arguments.orderedKeyEncoded(),
          configuration: JSONSchemaConfiguration(
            anyWhitespace: false,
            separators: .init(comma: ",", colon: ":"),
            isStrict: true
          )
        )
      else {
        preconditionFailure("Edge tool arguments must produce a valid JSON Schema.")
      }
      return grammar
    }

    static func qwenXMLArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      let schema = tool.arguments.orderedKeyEncoded()
      let structuralTag =
        #"{"type":"structural_tag","format":{"type":"json_schema","json_schema":\#(schema),"style":"qwen_xml","any_order":false}}"#
      guard let grammar = try? XGRGrammar.structuralTagJSON(structuralTag) else {
        preconditionFailure("Edge tool arguments must produce a valid Qwen XML structural tag.")
      }
      return grammar
    }

    static func toolCalls(
      tools: [EdgeToolDefinition],
      separator: String,
      range: GrammarToolCallRange,
      call: (EdgeToolDefinition) throws -> XGRGrammar
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw XGRError.invalidToolInvocationRange }
      guard let firstTool = tools.first else {
        guard range.lowerBound == 0 else {
          throw XGRError(
            code: EdgeToolsXGRError.emptyToolCollection,
            message: "A nonzero tool invocation range requires at least one tool."
          )
        }
        return try XGRGrammar.literal("")
      }

      var grammar = try call(firstTool)
      for tool in tools.dropFirst() {
        grammar = try grammar.union(call(tool))
      }
      return try Self.repeatingToolCall(grammar, separator: separator, range: range)
    }

    static func repeatingToolCall(
      _ call: XGRGrammar,
      separator: String,
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw XGRError.invalidToolInvocationRange }
      let separatedCall = try XGRGrammar.literal(separator).concatenate(call)

      switch range {
      case .exact(let count):
        return try Self.boundedToolCalls(
          call,
          separatedCall: separatedCall,
          minimum: count,
          maximum: count
        )
      case .bounded(let bounds):
        return try Self.boundedToolCalls(
          call,
          separatedCall: separatedCall,
          minimum: bounds.lowerBound,
          maximum: bounds.upperBound
        )
      case .unbounded(let minimum):
        guard minimum >= 0 else { throw XGRError.invalidToolInvocationRange }
        let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...)
        let nonempty = try call.concatenate(tail)
        return minimum == 0 ? try nonempty.optional() : nonempty
      }
    }

    private static func boundedToolCalls(
      _ call: XGRGrammar,
      separatedCall: XGRGrammar,
      minimum: Int,
      maximum: Int
    ) throws -> XGRGrammar {
      guard minimum >= 0, maximum >= minimum else { throw XGRError.invalidToolInvocationRange }
      guard maximum > 0 else { return try XGRGrammar.literal("") }
      let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...(maximum - 1))
      let nonempty = try call.concatenate(tail)
      return minimum == 0 ? try nonempty.optional() : nonempty
    }
  }

  // MARK: - Matcher Pool

  final class XGRToolCallMatcherPool {
    private final class CachedMatcher {
      private let matcher: XGRMatcher

      init(_ matcher: consuming XGRMatcher) {
        self.matcher = consume matcher
      }

      func fork() -> XGRMatcher {
        self.matcher.fork()
      }
    }

    private let maxCount: Int
    private var entries = [String: CachedMatcher]()
    private var order = [String]()

    init(maxCount: Int = 8) {
      self.maxCount = maxCount
    }

    func matcher(
      grammar: XGRGrammar,
      compilingWith compiler: borrowing XGRCompiler
    ) throws -> XGRMatcher {
      let key = grammar.ebnf
      if let cached = self.entries[key] {
        self.touch(key)
        return cached.fork()
      }
      let compiledGrammar = try compiler.compile(grammar)
      let matcher = try XGRMatcher(compiledGrammar: compiledGrammar)
      return self.insert(key, matcher: consume matcher)
    }

    func clear() {
      self.entries.removeAll()
      self.order.removeAll()
    }

    private func touch(_ key: String) {
      self.order.removeAll { $0 == key }
      self.order.append(key)
    }

    private func insert(
      _ key: String,
      matcher: consuming XGRMatcher
    ) -> XGRMatcher {
      let cached = CachedMatcher(consume matcher)
      if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
        self.entries.removeValue(forKey: leastRecentlyUsed)
        self.order.removeFirst()
      }
      self.entries[key] = cached
      self.order.append(key)
      return cached.fork()
    }
  }
#endif
