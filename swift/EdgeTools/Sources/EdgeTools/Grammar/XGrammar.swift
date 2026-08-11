#if XGrammar
  import EdgeToolsXGrammar
  import OrderedCollections

  extension XGRError.Code {
    /// An invalid ``GrammarToolCallRange`` for generation.
    public static let invalidToolInvocationRange = Self(
      rawValue: "invalid-tool-invocation-range"
    )

    /// An empty tool collection for generation.
    public static let emptyToolCollection = Self(rawValue: "empty-tool-collection")

    /// An unsupported tool schema for generation.
    public static let unsupportedToolSchema = Self(rawValue: "unsupported-tool-schema")

    /// An incompatible tokenizer vocabulary for generation.
    public static let incompatibleTokenizerVocabulary = Self(
      rawValue: "incompatible-tokenizer-vocabulary"
    )

    /// An invalid Needle tokenizer for generation.
    public static let invalidNeedleTokenizer = Self(rawValue: "invalid-needle-tokenizer")
  }

  extension XGRError {
    static let invalidHuggingFaceMetadata = Self(
      code: .invalidHuggingFaceMetadata,
      message: "Invalid Hugging Face tokenizer metadata."
    )
    static let invalidToolInvocationRange = Self(
      code: XGRError.Code.invalidToolInvocationRange,
      message: "Tool invocation ranges cannot have a negative lower bound."
    )
    static let unsupportedToolSchema = Self(
      code: XGRError.Code.unsupportedToolSchema,
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
        case .object(let decodedMetadata) = try EdgeToolsValue(json: Array(metadata.utf8)),
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

  extension XGRMatcher: EdgeToolsGrammarMatcher {}

  // MARK: - XGrammarEngine

  public final class XGrammarEngine: EdgeToolsGrammarEngine {
    private struct State {
      let compiler: XGRCompiler
      let matcherPool: XGRToolCallMatcherPool
    }

    public let tokenizerInfo: XGRTokenizerInfo
    private let state: Lock<State>

    public init(tokenizerInfo: XGRTokenizerInfo) throws {
      self.tokenizerInfo = tokenizerInfo
      self.state = try Lock {
        State(
          compiler: try XGRCompiler(tokenizerInfo: tokenizerInfo),
          matcherPool: XGRToolCallMatcherPool()
        )
      }
    }

    public func matcher(
      for grammar: XGRGrammar,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> XGRMatcher {
      try self.state.withLock { state in
        try state.matcherPool.matcher(
          grammar: grammar,
          compilingWith: state.compiler,
          stopTokenIds: stopTokenIds
        )
      }
    }

    public func clearCaches() {
      self.state.withLock { state in
        state.matcherPool.clear()
        state.compiler.clearCache()
      }
    }
  }

  // MARK: - Grammar

  extension XGRGrammar {
    public static func literal(_ literal: String) throws -> XGRGrammar {
      let escapedLiteral = literal.reduce(into: "") { result, character in
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
      return try Self.ebnf("root ::= \"\(escapedLiteral)\"")
    }

    public static func schema(_ schema: EdgeToolsGenerationSchema) -> XGRGrammar {
      guard let grammar = try? Self.jsonSchema(schema.orderedJSONString()) else {
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

  // MARK: - XGRGenerationConstraint

  public struct XGRGenerationConstraint: Sendable {
    private enum Kind: Sendable {
      case unconstrained
      case grammar(XGRGrammar)
      case toolsWithGrammar(
        range: GrammarToolCallRange = .unbounded(minimum: 0),
        grammar: (@Sendable (XGRGrammar, XGRTokenizerInfo) throws -> XGRGrammar)? = nil
      )
    }

    private let kind: Kind

    public static let unconstrained = Self(kind: .unconstrained)

    public static func grammar(_ grammar: XGRGrammar) -> Self {
      Self(kind: .grammar(grammar))
    }

    public static func toolsWithGrammar(
      range: GrammarToolCallRange = .unbounded(minimum: 0),
      grammar: (@Sendable (XGRGrammar, XGRTokenizerInfo) throws -> XGRGrammar)? = nil
    ) -> Self {
      Self(kind: .toolsWithGrammar(range: range, grammar: grammar))
    }

    public static let tools = Self.toolsWithGrammar()

    public static func toolsOrGrammar(
      _ userGrammar: XGRGrammar,
      range: GrammarToolCallRange = .unbounded(minimum: 0),
      transform: (@Sendable (XGRGrammar, XGRTokenizerInfo) throws -> XGRGrammar)? = nil
    ) -> Self {
      toolsWithGrammar(
        range: range,
        grammar: { toolsGrammar, tokenizerInfo in
          let resolvedToolsGrammar =
            try transform?(toolsGrammar, tokenizerInfo) ?? toolsGrammar
          return try resolvedToolsGrammar.union(userGrammar)
        }
      )
    }

    public static func schema(_ type: (some EdgeToolsGenerable).Type) -> Self {
      Self.grammar(.schema(type))
    }
  }

  extension XGRGenerationConstraint: EdgeToolsGenerationConstraint {
    public typealias Context = XGrammarEngine

    public var toolCallRange: GrammarToolCallRange? {
      guard case .toolsWithGrammar(let range, _) = self.kind else { return nil }
      return range
    }

    public func grammar(
      toolCallGrammar: XGRGrammar?,
      context: XGrammarEngine
    ) throws -> XGRGrammar {
      switch self.kind {
      case .unconstrained:
        return .universal
      case .grammar(let grammar):
        return grammar
      case .toolsWithGrammar(_, let transform):
        let grammar = toolCallGrammar ?? .universal
        return try transform?(grammar, context.tokenizerInfo) ?? grammar
      }
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
        if let separator = "::=".firstRange(in: line) {
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

    var ruleReferences: [String: [String]] {
      self.ruleContents { body, token in
        token.kind == .identifier ? String(body[token.range]) : nil
      }
    }

    var ruleLiterals: [String: [String]] {
      self.ruleContents { body, token in
        guard token.kind == .literal else { return nil }
        return Self.decodeLiteral(body[token.range].dropFirst().dropLast())
      }
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

    mutating func mapRuleReferences(
      _ transform: (_ ruleName: String, _ reference: String) -> String
    ) {
      for index in self.rules.indices {
        let rule = self.rules[index]
        var output = ""
        var outputStart = rule.body.startIndex
        for token in rule.body.ebnfTokens where token.kind == .identifier {
          output.append(contentsOf: rule.body[outputStart..<token.range.lowerBound])
          output.append(transform(rule.name, String(rule.body[token.range])))
          outputStart = token.range.upperBound
        }
        output.append(contentsOf: rule.body[outputStart...])
        self.rules[index].body = output
      }
    }

    private static func mapLiterals(
      in source: String,
      _ transform: (_ value: String, _ suffix: Substring) -> String
    ) throws -> String {
      var output = ""
      var outputStart = source.startIndex

      for token in source.ebnfTokens where token.kind == .literal {
        output.append(contentsOf: source[outputStart..<token.range.lowerBound])
        let encodedValue = source[token.range].dropFirst().dropLast()
        let value = Self.decodeLiteral(encodedValue)
        let transformed = transform(value, source[token.range.upperBound...])
        output.append("\"")
        output.append(contentsOf: Self.escapeLiteral(transformed))
        output.append("\"")
        outputStart = token.range.upperBound
      }
      output.append(contentsOf: source[outputStart...])
      return output
    }

    private static func replacingRuleReferences(in source: String, aliases: [String: String])
      -> String
    {
      var output = ""
      var outputStart = source.startIndex
      for token in source.ebnfTokens where token.kind == .identifier {
        guard let replacement = aliases[String(source[token.range])] else { continue }
        output.append(contentsOf: source[outputStart..<token.range.lowerBound])
        output.append(replacement)
        outputStart = token.range.upperBound
      }
      output.append(contentsOf: source[outputStart...])
      return output
    }

    private func ruleContents(
      _ transform: (_ body: String, _ token: EBNFToken) -> String?
    ) -> [String: [String]] {
      Dictionary(
        self.rules.map { rule in
          (rule.name, rule.body.ebnfTokens.compactMap { transform(rule.body, $0) })
        },
        uniquingKeysWith: +
      )
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
    func hasRuleReference(withPrefix prefix: String) -> Bool {
      self.ebnfTokens.contains { token in
        token.kind == .identifier && self[token.range].hasPrefix(prefix)
      }
    }

    var hasToolCallContinuationReference: Bool {
      self.ebnfTokens.contains { token in
        token.kind == .identifier && self[token.range].isToolCallContinuationName
      }
    }
  }

  // MARK: - Tool Call Building

  extension XGRGrammar {
    static func strictJSONArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      guard
        let grammar = try? XGRGrammar.jsonSchema(
          tool.arguments.orderedJSONString(),
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

    static func xmlToolArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      let schema = tool.arguments.orderedJSONString()
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
            code: XGRError.Code.emptyToolCollection,
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
    private let maxCount: Int
    private var entries = [String: XGRMatcher]()
    private var order = [String]()

    init(maxCount: Int = 8) {
      self.maxCount = maxCount
    }

    func matcher(
      grammar: XGRGrammar,
      compilingWith compiler: XGRCompiler,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> XGRMatcher {
      let sortedStopTokenIds = stopTokenIds.sorted()
      let key = "\(grammar.ebnf)\u{0}\(sortedStopTokenIds.map(String.init).joined(separator: ","))"
      if let cached = self.entries[key] {
        self.touch(key)
        return cached.fork()
      }
      let compiledGrammar = try compiler.compile(grammar)
      let matcher = try XGRMatcher(
        compiledGrammar: compiledGrammar,
        overrideStopTokenIDs: sortedStopTokenIds
      )
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
      matcher: XGRMatcher
    ) -> XGRMatcher {
      if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
        self.entries.removeValue(forKey: leastRecentlyUsed)
        self.order.removeFirst()
      }
      self.entries[key] = matcher
      self.order.append(key)
      return matcher.fork()
    }
  }

  // MARK: - EBNF Scanning

  private struct EBNFToken {
    enum Kind {
      case literal
      case identifier
    }

    let kind: Kind
    let range: Range<String.Index>
  }

  extension StringProtocol where Index == String.Index, SubSequence == Substring {
    fileprivate var ebnfTokens: [EBNFToken] {
      var tokens = [EBNFToken]()
      var index = self.startIndex
      while index < self.endIndex {
        let character = self[index]
        if character == "\"" {
          var end = self.index(after: index)
          var isTerminated = false
          while end < self.endIndex {
            if self[end] == "\\" {
              end = self.index(after: end)
              guard end < self.endIndex else { break }
            } else if self[end] == "\"" {
              isTerminated = true
              end = self.index(after: end)
              break
            }
            end = self.index(after: end)
          }
          if isTerminated {
            tokens.append(EBNFToken(kind: .literal, range: index..<end))
          }
          index = end
        } else if character.isEBNFWordCharacter {
          var end = index
          while end < self.endIndex, self[end].isEBNFWordCharacter {
            end = self.index(after: end)
          }
          // NB: A digit-led run is a whole word, so an identifier can never start part way into it.
          if character.isEBNFIdentifierStart {
            tokens.append(EBNFToken(kind: .identifier, range: index..<end))
          }
          index = end
        } else {
          // NB: Character classes such as [^\0-\x1f\"\\] contain escaped quotes, so a backslash has
          // to consume the character after it or an escaped quote would open a bogus literal.
          if character == "\\" {
            index = self.index(after: index)
            guard index < self.endIndex else { break }
          }
          index = self.index(after: index)
        }
      }
      return tokens
    }

    fileprivate var isToolCallContinuationName: Bool {
      var suffix: Substring
      if self.starts(with: "xml_object") {
        suffix = self.dropFirst("xml_object".count)
      } else if self.starts(with: "root") {
        suffix = self.dropFirst("root".count)
      } else {
        return false
      }
      while !suffix.isEmpty {
        guard suffix.first == "_" else { return false }
        suffix = suffix.dropFirst()
        let segment = suffix.prefix { $0.isASCII && ($0.isLetter || $0.isNumber) }
        guard !segment.isEmpty else { return false }
        suffix = suffix[segment.endIndex...]
      }
      return true
    }
  }

  extension Character {
    fileprivate var isEBNFIdentifierStart: Bool {
      self == "_" || (self.isASCII && self.isLetter)
    }

    fileprivate var isEBNFWordCharacter: Bool {
      self.isEBNFIdentifierStart || (self.isASCII && self.isNumber)
    }
  }
#endif
