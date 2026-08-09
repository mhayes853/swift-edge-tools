import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  // MARK: - MiniCPM5 Model

  public struct MiniCPM5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsConversationalPrompt
    public typealias GenerationParser = MiniCPM5GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func grammar(
      prompt: EdgeToolsConversationalPrompt,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, context: context) { range in
        let toolCalls = try XGRGrammar.miniCPM5(tools: tools, range: range)
        guard prompt.reasoningEffort.isEnabled else { return toolCalls }
        return try XGRGrammar.qwenReasoning().concatenate(toolCalls)
      }
    }

    public static func templateContext(prompt: EdgeToolsConversationalPrompt) -> [String: any Sendable]? {
      guard prompt.reasoningEffort != .default else { return nil }
      return ["enable_thinking": prompt.reasoningEffort.isEnabled]
    }

    public static func prepare(
      prompt: inout EdgeToolsConversationalPrompt,
      tools: [EdgeToolDefinition],
      parser: inout MiniCPM5GenerationParser
    ) {
      let prefix = prompt.reasoningEffort.isEnabled ? "<think>\n" : "<think>\n\n</think>\n\n"
      _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
    }
  }

  public typealias MiniCPM5MLXModelEngine = MLXEngine<MiniCPM5MLXProfile>
#endif

// MARK: - MiniCPM5 Tool Call Parsing

public struct MiniCPM5GenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<function",
    toolCloser: "</function>",
    reasoningOpener: "<think>",
    reasoningCloser: "</think>",
    ignoredToolRegions: [.init(opener: "<![CDATA[", closer: "]]>")],
    parseToolCalls: MiniCPM5ToolCalls.parse
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}

private enum MiniCPM5ToolCalls {
  private static let cdataOpener = Array("<![CDATA[".utf8)
  private static let cdataCloser = Array("]]>".utf8)
  private static let parameterCloser = Array("</param>".utf8)

  static func parse(_ source: String) -> [EdgeRawToolCall] {
    var block = IncrementalToolCallBlock(opener: "<function", closer: "</function>")
    block.append(EdgeToolsToken(id: -1, stringValue: source))
    var calls = [EdgeRawToolCall]()
    while let payload = block.nextPayload(
      outsideRegionOpenedBy: Self.cdataOpener,
      closedBy: Self.cdataCloser,
      orAbortedBy: Self.parameterCloser
    ) {
      let source = String(decoding: payload, as: UTF8.self)
      if let call = Self.parseCall(source) { calls.append(call) }
    }
    return calls
  }

  private static func parseCall(_ source: String) -> EdgeRawToolCall? {
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
