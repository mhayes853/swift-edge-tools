import EdgeToolsCore
import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  // MARK: - MiniCPM5 Model

  public struct MiniCPM5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = MiniCPM5GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      let grammar = try Self.constrainedGrammar(
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      ) {
        try XGRGrammar.miniCPM5(tools: tools, range: $0)
      }
      guard prompt.reasoningEffort.isEnabled else { return grammar }
      return try .qwenReasoning().concatenate(grammar)
    }

    public static func templateContext(prompt: EdgeToolsTranscript) -> [String: EdgeToolsValue]? {
      guard prompt.reasoningEffort != .default else { return nil }
      return ["enable_thinking": .boolean(prompt.reasoningEffort.isEnabled)]
    }

    public static func prepare(
      prompt: inout EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parser: inout MiniCPM5GenerationParser
    ) {
      let prefix = prompt.reasoningEffort.isEnabled ? "<think>\n" : "<think>\n\n</think>\n\n"
      _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
    }
  }

  public typealias MiniCPM5MLXModelEngine = MLXEngine<MiniCPM5MLXProfile>
#endif

#if Llama && XGrammar && canImport(CLlama)
  public struct MiniCPM5LlamaProfile: LlamaModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = MiniCPM5GenerationParser
    public typealias GenerateParameters = DefaultLlamaGenerateParameters
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultLlamaGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      ) { range in
        let toolCalls = try XGRGrammar.miniCPM5(tools: tools, range: range)
        guard prompt.reasoningEffort.isEnabled else { return toolCalls }
        return try .qwenReasoning().concatenate(toolCalls)
      }
    }

    public static func templateContext(prompt: EdgeToolsTranscript) -> [String: EdgeToolsValue]? {
      guard prompt.reasoningEffort != .default else { return nil }
      return ["enable_thinking": .boolean(prompt.reasoningEffort.isEnabled)]
    }

    public static func prepare(
      prompt: inout EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parser: inout MiniCPM5GenerationParser
    ) {
      let prefix = prompt.reasoningEffort.isEnabled ? "<think>\n" : "<think>\n\n</think>\n\n"
      _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
    }
  }

  public typealias MiniCPM5LlamaModelEngine = LlamaEngine<MiniCPM5LlamaProfile>
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
        guard isMiniCPM5TopLevelArgumentRule(ruleName) else { return value }
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

      // NB: Enum, pattern, union, and nullable strings are emitted as their own rules rather than
      // inline in the argument rule, so the raw rewrite has to follow those references too.
      let valueRules = miniCPM5ParameterValueRules(in: document)
      try document.mapLiterals { ruleName, value, _ in
        guard valueRules.contains(ruleName) else { return value }
        if value == "\"" { return "" }
        guard value.count >= 2, value.first == "\"", value.last == "\"" else { return value }
        return String(value.dropFirst().dropLast())
      }
      document.mapRuleReferences { ruleName, reference in
        guard reference == "basic_string",
          isMiniCPM5TopLevelArgumentRule(ruleName) || valueRules.contains(ruleName)
        else { return reference }
        return "mini_cpm_string"
      }
      document.rules.append(contentsOf: miniCPM5StringRules)

      let arguments = try XGRGrammar.ebnf(document.source)
      let prefix = try XGRGrammar.literal("<function name=\"\(tool.name)\">")
      return
        try prefix
        .concatenate(arguments)
        .concatenate(XGRGrammar.literal("</function>"))
    }
  }

  // MARK: - MiniCPM5 Raw Parameter Values

  // NB: MiniCPM5's own chat template documents parameter values as raw text, escaped through CDATA
  // when they contain "<", "&", or a newline. Values are therefore unquoted, which makes a string
  // ambiguous with the JSON primitive it happens to spell, so a raw value may not begin with a
  // character that starts a JSON value, nor spell "true", "false", or "null" on its own.
  private let miniCPM5RawTail = #"[^<&\r\n]"#
  private let miniCPM5RawHead = #"[^<&\r\n \t\"\-0-9{\[tfn]"#
  private let miniCPM5PrimitiveWords = ["true", "false", "null"]

  private var miniCPM5StringRules: [XGREBNFDocument.Rule] {
    let alternatives =
      ["mini_cpm_raw_plain"] + miniCPM5PrimitiveWords.map { "mini_cpm_raw_\($0.first!)" }
    return [
      XGREBNFDocument.Rule(
        name: "mini_cpm_string",
        body: #"mini_cpm_raw | ("<![CDATA[" mini_cpm_cdata "]]>")"#
      ),
      XGREBNFDocument.Rule(name: "mini_cpm_raw", body: alternatives.joined(separator: " | ")),
      XGREBNFDocument.Rule(name: "mini_cpm_raw_plain", body: "\(miniCPM5RawHead) mini_cpm_tail"),
      XGREBNFDocument.Rule(
        name: "mini_cpm_tail",
        body: #"("") | (\#(miniCPM5RawTail) mini_cpm_tail)"#
      ),
      // NB: A trailing run of whitespace is trimmed off before parsing, so a value that spells a
      // primitive followed only by spaces has to be rejected alongside the bare primitive.
      XGREBNFDocument.Rule(
        name: "mini_cpm_raw_suffix",
        body: #"mini_cpm_space [^<&\r\n \t] mini_cpm_tail"#
      ),
      XGREBNFDocument.Rule(name: "mini_cpm_space", body: #"("") | ([ \t] mini_cpm_space)"#),
      XGREBNFDocument.Rule(
        name: "mini_cpm_cdata",
        body: #"("") | (mini_cpm_cdata_char mini_cpm_cdata)"#
      ),
      XGREBNFDocument.Rule(name: "mini_cpm_cdata_char", body: #"[^\]] | ("]" [^\]]) | ("]]" [^>])"#)
    ] + miniCPM5PrimitiveWords.flatMap(miniCPM5RawWordRules)
  }

  // Emits the rules matching every raw value that starts with `word`'s first character but never
  // spells `word` itself, by branching away at each character that diverges from it.
  private func miniCPM5RawWordRules(excluding word: String) -> [XGREBNFDocument.Rule] {
    let characters = Array(word)
    let name = "mini_cpm_raw_\(characters[0])"
    let chain = characters.indices.dropFirst()
      .map { index -> XGREBNFDocument.Rule in
        let character = characters[index]
        let next = index == characters.count - 1 ? "mini_cpm_raw_suffix" : "\(name)_\(index + 1)"
        return XGREBNFDocument.Rule(
          name: "\(name)_\(index)",
          body: #"("") | ([^<&\r\n\#(character)] mini_cpm_tail) | ("\#(character)" \#(next))"#
        )
      }
    return [XGREBNFDocument.Rule(name: name, body: #""\#(characters[0])" \#(name)_1"#)] + chain
  }

  // Walks out from the argument rules to the rules holding each parameter's value, stopping at any
  // value that opens a JSON container since its nested strings stay conventionally quoted.
  private func miniCPM5ParameterValueRules(in document: XGREBNFDocument) -> Set<String> {
    let references = document.ruleReferences
    let literals = document.ruleLiterals
    var names = Set<String>()
    var pending = references.filter { isMiniCPM5TopLevelArgumentRule($0.key) }
      .values
      .flatMap { $0 }

    while let name = pending.popLast() {
      guard !name.hasPrefix("basic_"), !isMiniCPM5TopLevelArgumentRule(name),
        !names.contains(name),
        !(literals[name] ?? []).contains(where: { $0 == "{" || $0 == "[" }),
        !(references[name] ?? [])
          .contains(where: {
            ["basic_object", "basic_array", "basic_any"].contains($0)
          })
      else { continue }
      names.insert(name)
      pending.append(contentsOf: references[name] ?? [])
    }
    return names
  }

  private func isMiniCPM5TopLevelArgumentRule(_ name: String) -> Bool {
    guard name != "root" else { return true }
    guard name.starts(with: "root_") else { return false }
    var digits = name.dropFirst("root_".count)
    if digits.starts(with: "part_") { digits = digits.dropFirst("part_".count) }
    return !digits.isEmpty && digits.allSatisfy { $0.isASCII && $0.isNumber }
  }
#endif
