// MARK: - EBNF

#if XGrammar
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
#endif

// MARK: - Tool Call Building

#if XGrammar
  extension XGRGrammar {
    static func strictJSONArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      guard
        let grammar = try? XGRGrammar(
          jsonSchema: tool.arguments.orderedKeyEncoded(),
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
      guard let grammar = try? XGRGrammar(structuralTagJSON: structuralTag) else {
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
            code: .emptyToolCollection,
            message: "A nonzero tool invocation range requires at least one tool."
          )
        }
        return try XGRGrammar(literal: "")
      }

      var grammar = try call(firstTool)
      for tool in tools.dropFirst() {
        grammar = try grammar.union(call(tool))
      }
      return try Self.repeatingToolCall(grammar, separator: separator, range: range)
    }

    static func repeatingToolCall(
      _ call: borrowing XGRGrammar,
      separator: String,
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw XGRError.invalidToolInvocationRange }
      let separatedCall = try XGRGrammar(literal: separator).concatenate(call)

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
      _ call: borrowing XGRGrammar,
      separatedCall: borrowing XGRGrammar,
      minimum: Int,
      maximum: Int
    ) throws -> XGRGrammar {
      guard minimum >= 0, maximum >= minimum else {
        throw XGRError.invalidToolInvocationRange
      }
      guard maximum > 0 else { return try XGRGrammar(literal: "") }
      let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...(maximum - 1))
      let nonempty = try call.concatenate(tail)
      return minimum == 0 ? try nonempty.optional() : nonempty
    }
  }

#endif

// MARK: - Matcher Pool

#if XGrammar
  final class XGRToolCallMatcherPool {
    typealias NormalizeTools = ([EdgeToolDefinition]) -> [EdgeToolDefinition]
    typealias MakeGrammar =
      ([EdgeToolDefinition], GrammarToolCallRange) throws -> XGRGrammar

    private final class CachedMatcher {
      private let matcher: XGRMatcher

      init(_ matcher: consuming XGRMatcher) {
        self.matcher = consume matcher
      }

      func fork() -> XGRMatcher {
        self.matcher.fork()
      }
    }

    private struct Key: Hashable, Sendable {
      let tools: [EdgeToolDefinition]
      let range: GrammarToolCallRange
    }

    private let maxCount: Int
    private let normalizeTools: NormalizeTools
    private let makeGrammar: MakeGrammar
    private var entries = [Key: CachedMatcher]()
    private var order = [Key]()

    init(
      maxCount: Int = 8,
      normalizeTools: @escaping NormalizeTools = { $0 },
      makeGrammar: @escaping MakeGrammar
    ) {
      self.maxCount = maxCount
      self.normalizeTools = normalizeTools
      self.makeGrammar = makeGrammar
    }

    func matcher(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange,
      compilingWith compiler: borrowing XGRCompiler
    ) throws -> XGRMatcher {
      let key = Key(tools: self.normalizeTools(Array(tools)), range: range)
      if let cached = self.entries[key] {
        self.touch(key)
        return cached.fork()
      }
      let grammar = try self.makeGrammar(key.tools, key.range)
      let compiledGrammar = try compiler.compile(grammar)
      let matcher = try XGRMatcher(compiledGrammar: compiledGrammar)
      return self.insert(key, matcher: consume matcher)
    }

    func clear() {
      self.entries.removeAll()
      self.order.removeAll()
    }

    private func touch(_ key: Key) {
      self.order.removeAll { $0 == key }
      self.order.append(key)
    }

    private func insert(
      _ key: Key,
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
