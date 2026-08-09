#if XGrammar
  import EdgeToolsXGrammar
#endif

// MARK: - Gemma Tool Call Format

package struct GemmaToolCallFormat: Hashable, Sendable {
  package var opener: String
  package var closer: String
  package var stringMarker: String
  package var marksAllValues: Bool

  package static var functionGemma: Self {
    Self(
      opener: "<start_function_call>",
      closer: "<end_function_call>",
      stringMarker: "<escape>",
      marksAllValues: true
    )
  }

  package static var gemma4: Self {
    Self(
      opener: "<|tool_call>",
      closer: "<tool_call|>",
      stringMarker: "<|\"|>",
      marksAllValues: false
    )
  }
}

// MARK: - Gemma Tool Call Parsers

public struct Gemma4GenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<|tool_call>",
    toolCloser: "<tool_call|>",
    reasoningOpener: "<|channel>thought\n",
    reasoningCloser: "<channel|>",
    parseToolCalls: { GemmaToolCalls.parse($0, format: .gemma4) }
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}

enum GemmaToolCalls {
  static func parse(_ source: String, format: GemmaToolCallFormat) -> [EdgeRawToolCall] {
    var block = IncrementalToolCallBlock(opener: format.opener, closer: format.closer)
    block.append(EdgeToolsToken(id: -1, stringValue: source))
    var calls = [EdgeRawToolCall]()
    let marker = Array(format.stringMarker.utf8)
    while let payloadData = block.nextPayload(outside: marker) {
      let payload = String(decoding: payloadData, as: UTF8.self)
      var reader = GemmaCallReader(
        source: payload,
        stringMarker: format.stringMarker,
        markedValuesAreJSON: format.marksAllValues
      )
      if let call = reader.parse() { calls.append(call) }
    }
    return calls
  }
}

// MARK: - Gemma Call Reader

private struct GemmaCallReader: ToolCallValueReader {
  var cursor: ToolCallStringCursor
  let stringMarker: String
  let markedValuesAreJSON: Bool

  init(source: String, stringMarker: String, markedValuesAreJSON: Bool) {
    self.cursor = ToolCallStringCursor(source)
    self.stringMarker = stringMarker
    self.markedValuesAreJSON = markedValuesAreJSON
  }

  mutating func parse() -> EdgeRawToolCall? {
    self.cursor.skipWhitespace()
    guard self.cursor.consume("call:") else { return nil }
    guard let name = self.cursor.read(until: "{")?.trimmingWhitespace,
      !name.isEmpty,
      self.cursor.consume("{")
    else { return nil }

    self.cursor.skipWhitespace()
    guard let arguments = self.parseObjectBody() else { return nil }
    self.cursor.skipWhitespace()
    guard self.cursor.isAtEnd else { return nil }
    return EdgeRawToolCall(name: name, arguments: .object(arguments))
  }

  mutating func parseObjectKey() -> String? {
    guard
      let source = self.cursor.read(until: ":")?.trimmingWhitespace,
      !source.isEmpty,
      self.cursor.consume(":")
    else { return nil }
    if source.count >= 2, source.first == "\"", source.last == "\"",
      case .string(let key) = try? EdgeToolsValue(json: source)
    {
      return key
    }
    return source
  }

  mutating func parseValue() -> EdgeToolsValue? {
    self.cursor.skipWhitespace()
    if self.cursor.remainder.hasPrefix(self.stringMarker) {
      return self.parseMarkedValue()
    }
    guard let character = self.cursor.current else { return nil }
    switch character {
    case "{":
      self.cursor.advance()
      return self.parseObjectBody().map(EdgeToolsValue.object)
    case "[": return self.parseArray()
    default: return Self.parseBareValue(self.readBareValue())
    }
  }

  private mutating func parseMarkedValue() -> EdgeToolsValue? {
    guard self.cursor.consume(self.stringMarker) else { return nil }
    guard let value = self.cursor.read(until: self.stringMarker) else { return nil }
    guard self.cursor.consume(self.stringMarker) else { return nil }
    if self.markedValuesAreJSON, let decoded = try? EdgeToolsValue(json: value) {
      return decoded
    }
    return .string(value)
  }

  private mutating func readBareValue() -> String {
    self.cursor.read { ![",", "}", "]"].contains($0) }.trimmingWhitespace
  }

  private static func parseBareValue(_ token: String) -> EdgeToolsValue? {
    if token.isEmpty {
      return nil
    }
    if let value = parseToolCallBooleanOrNull(token) {
      return value
    }
    if let integer = Int(token) {
      return .integer(integer)
    }
    if let number = Double(token) {
      return .number(number)
    }
    return .string(token)
  }
}

// MARK: - Gemma Tool Call Grammars

#if XGrammar
  extension XGRGrammar {
    public static func functionGemma(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.gemmaToolCalls(tools: tools, range: range, format: .functionGemma)
    }

    public static func gemma4(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.gemmaToolCalls(tools: tools, range: range, format: .gemma4)
    }

    private static func gemmaToolCalls(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange,
      format: GemmaToolCallFormat
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.gemmaToolCall($0, format: format)
      }
    }

    private static func gemmaToolCall(
      _ tool: EdgeToolDefinition,
      format: GemmaToolCallFormat
    ) throws -> XGRGrammar {
      let xmlArguments = Self.xmlToolArguments(for: tool)
      var document = try XGREBNFDocument(xmlArguments.ebnf)
      try document.mapLiterals { ruleName, value, suffix in
        if value == "</parameter>" {
          if ruleName.hasPrefix("xml_string") {
            return format.stringMarker
          }
          let marker = format.marksAllValues ? format.stringMarker : ""
          return suffix.hasToolCallContinuationReference ? "\(marker)," : marker
        }
        if value.hasPrefix("<parameter="), value.hasSuffix(">") {
          let name = value.dropFirst("<parameter=".count).dropLast()
          let marker =
            if format.marksAllValues || suffix.hasRuleReference(withPrefix: "xml_string") {
              format.stringMarker
            } else {
              ""
            }
          return "\(name):\(marker)"
        }
        if value == "<parameter=" {
          return ""
        }
        if value == ">", ruleName.hasPrefix("xml_object") {
          return format.marksAllValues ? ":\(format.stringMarker)" : ":"
        }
        return value
      }
      if !format.marksAllValues {
        document.mapRuleReferences { ruleName, reference in
          guard reference.hasPrefix("xml_string"), !ruleName.hasPrefix("xml_string") else {
            return reference
          }
          return "\(reference) \"<|\\\"|>\""
        }
      }

      let arguments = try XGRGrammar.ebnf(document.source)
      let prefix = try XGRGrammar.literal("\(format.opener)call:\(tool.name){")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar.literal("}\(format.closer)"))
    }
  }
#endif
