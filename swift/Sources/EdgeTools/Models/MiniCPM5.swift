import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  // MARK: - MiniCPM5 Model

  public struct MiniCPM5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = MiniCPM5ToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .miniCPM5(tools: tools, range: range)
    }
  }

  public typealias MiniCPM5MLXModelEngine = MLXEngine<MiniCPM5MLXProfile>
#endif

// MARK: - MiniCPM5 Tool Call Parsing

public struct MiniCPM5ToolCallParser: EdgeToolCallParser, Sendable {
  private static let cdataOpener = Array("<![CDATA[".utf8)
  private static let cdataCloser = Array("]]>".utf8)
  private static let parameterCloser = Array("</param>".utf8)

  private var block = IncrementalToolCallBlock(opener: "<function", closer: "</function>")

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeRawToolCall] {
    self.block.append(token)
    var calls = [EdgeRawToolCall]()
    while let call = self.nextCall() {
      calls.append(call)
    }
    return calls
  }

  private mutating func nextCall() -> EdgeRawToolCall? {
    while let payload = self.block.nextPayload(
      outsideRegionOpenedBy: Self.cdataOpener,
      closedBy: Self.cdataCloser,
      orAbortedBy: Self.parameterCloser
    ) {
      let source = String(decoding: payload, as: UTF8.self)
      if let call = Self.parse(source) {
        return call
      }
    }
    return nil
  }

  private static func parse(_ source: String) -> EdgeRawToolCall? {
    var cursor = ToolCallStringCursor(source)
    cursor.skipWhitespace()
    guard cursor.consume("name="), let quote = cursor.current,
      quote == "\"" || quote == "'"
    else { return nil }
    cursor.advance()
    guard let name = cursor.read(until: String(quote)), !name.isEmpty,
      cursor.consume(character: quote)
    else { return nil }
    cursor.skipWhitespace()
    guard cursor.consume(">") else { return nil }
    var arguments = OrderedDictionary<String, EdgeToolsValue>()

    while true {
      cursor.skipWhitespace()
      if cursor.isAtEnd { break }
      guard cursor.consume("<param") else { return nil }
      cursor.skipWhitespace()
      guard cursor.consume("name="), let quote = cursor.current,
        quote == "\"" || quote == "'"
      else { return nil }
      cursor.advance()
      guard let parameterName = cursor.read(until: String(quote)), !parameterName.isEmpty,
        arguments[parameterName] == nil,
        cursor.consume(character: quote)
      else { return nil }
      cursor.skipWhitespace()
      guard cursor.consume(">"), let value = Self.parseParameterValue(cursor: &cursor) else {
        return nil
      }
      arguments[parameterName] = value
    }
    return EdgeRawToolCall(name: name, arguments: .object(arguments))
  }

  private static func parseParameterValue(
    cursor: inout ToolCallStringCursor
  ) -> EdgeToolsValue? {
    if cursor.consume("<![CDATA[") {
      guard let value = cursor.read(until: "]]>"), cursor.consume("]]>") else { return nil }
      guard cursor.consume("</param>") else { return nil }
      return .string(value)
    }

    guard let rawValue = cursor.read(until: "</param>"), cursor.consume("</param>") else {
      return nil
    }
    let source = rawValue.trimmingWhitespace
    if let value = parseToolCallBooleanOrNull(source) {
      return value
    }
    if let value = try? EdgeToolsValue(json: source) {
      return value
    }
    return .string(source)
  }
}

// MARK: - MiniCPM5 Grammar

#if XGrammar
  extension XGRGrammar {
    public static func miniCPM5(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "\n", range: range) {
        try Self.miniCPM5Call($0)
      }
    }

    private static func miniCPM5Call(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      var document = try XGREBNFDocument(Self.strictJSONArguments(for: tool).ebnf)
      try document.mapLiterals { ruleName, value, suffix in
        guard Self.isMiniCPM5TopLevelArgumentRule(ruleName) else { return value }
        switch value {
        case "{", "{}", ":":
          return ""
        case ",", "}":
          return "</param>"
        default:
          guard value.count >= 2, value.first == "\"", value.last == "\"",
            #"":""#.firstRange(in: suffix) != nil
          else { return value }
          return "<param name=\"\(value.dropFirst().dropLast())\">"
        }
      }
      document.mapRuleReferences { ruleName, reference in
        guard Self.isMiniCPM5TopLevelArgumentRule(ruleName), reference == "basic_string" else {
          return reference
        }
        return "mini_cpm_string"
      }
      document.rules.append(
        XGREBNFDocument.Rule(
          name: "mini_cpm_string",
          body: #"basic_string | ("<![CDATA[" ([^\]] | "]" [^\]] | "]]" [^>])* "]]>")"#
        )
      )

      let arguments = try XGRGrammar.ebnf(document.source)
      let prefix = try XGRGrammar.literal("<function name=\"\(tool.name)\">")
      return
        try prefix
        .concatenate(arguments)
        .concatenate(XGRGrammar.literal("</function>"))
    }

    private static func isMiniCPM5TopLevelArgumentRule(_ name: String) -> Bool {
      guard name != "root" else { return true }
      guard name.starts(with: "root_") else { return false }
      var digits = name.dropFirst("root_".count)
      if digits.starts(with: "part_") {
        digits = digits.dropFirst("part_".count)
      }
      return !digits.isEmpty && digits.allSatisfy { $0.isASCII && $0.isNumber }
    }
  }
#endif
