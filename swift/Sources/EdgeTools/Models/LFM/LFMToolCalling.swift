import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

// MARK: - LFM2PythonToolCallParser

public struct LFM2PythonToolCallParser: EdgeToolCallParser, Sendable {
  private var list = IncrementalToolCallList(opener: "<|tool_call_start|>")

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeRawToolCall] {
    self.list.append(token)
    var calls = [EdgeRawToolCall]()
    while let call = self.nextCall() {
      calls.append(call)
    }
    return calls
  }

  private mutating func nextCall() -> EdgeRawToolCall? {
    while let sourceData = self.list.nextItem(findRange: { $0.firstCompletePythonCallRange() }) {
      let source = String(decoding: sourceData, as: UTF8.self)
      var reader = PythonCallReader(source: source)
      if let call = reader.parse() {
        return call
      }
    }
    return nil
  }
}

// MARK: - LFM2

public typealias LFM2ToolCallParser = LFM2PythonToolCallParser
public typealias LFM2P5ToolCallParser = LFM2PythonToolCallParser

// MARK: - Python Call Boundaries

extension Array where Element == UInt8 {
  fileprivate func firstCompletePythonCallRange() -> Range<Int>? {
    var state = PythonCallBoundaryState()
    for index in self.indices {
      guard let isComplete = state.consume(self[index]) else { return nil }
      if isComplete { return 0..<(index + 1) }
    }
    return nil
  }
}

private struct PythonCallBoundaryState: Hashable, Sendable {
  private var parenthesisDepth = 0
  private var bracketDepth = 0
  private var braceDepth = 0
  private var quote: UInt8?
  private var isEscaping = false
  private var hasSeenParenthesis = false

  mutating func consume(_ byte: UInt8) -> Bool? {
    if let quote {
      self.consumeQuoted(byte, quote: quote)
      return false
    }
    self.consumeUnquoted(byte)
    guard self.isValid else { return nil }
    return self.isComplete
  }

  private var isValid: Bool {
    self.parenthesisDepth >= 0 && self.bracketDepth >= 0 && self.braceDepth >= 0
  }

  private var isComplete: Bool {
    self.hasSeenParenthesis && self.parenthesisDepth == 0
      && self.bracketDepth == 0 && self.braceDepth == 0
  }

  private mutating func consumeQuoted(_ byte: UInt8, quote: UInt8) {
    if self.isEscaping {
      self.isEscaping = false
    } else if byte == UInt8(ascii: "\\") {
      self.isEscaping = true
    } else if byte == quote {
      self.quote = nil
    }
  }

  private mutating func consumeUnquoted(_ byte: UInt8) {
    switch byte {
    case UInt8(ascii: "'"), UInt8(ascii: "\""):
      self.quote = byte
    case UInt8(ascii: "("):
      self.parenthesisDepth += 1
      self.hasSeenParenthesis = true
    case UInt8(ascii: ")"):
      self.parenthesisDepth -= 1
    case UInt8(ascii: "["):
      self.bracketDepth += 1
    case UInt8(ascii: "]"):
      self.bracketDepth -= 1
    case UInt8(ascii: "{"):
      self.braceDepth += 1
    case UInt8(ascii: "}"):
      self.braceDepth -= 1
    default:
      break
    }
  }
}

// MARK: - PythonCallReader

private struct PythonCallReader: ToolCallValueReader {
  private static let escapedCharacters: [Character: Character] = [
    "\\": "\\",
    "'": "'",
    "\"": "\"",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "b": "\u{8}",
    "f": "\u{c}"
  ]

  var cursor: ToolCallStringCursor

  init(source: String) {
    self.cursor = ToolCallStringCursor(source)
  }

  mutating func parse() -> EdgeRawToolCall? {
    self.cursor.skipWhitespace()
    guard let name = self.parseIdentifier(), self.cursor.consume("(") else { return nil }
    var arguments = OrderedDictionary<String, EdgeToolsValue>()
    self.cursor.skipWhitespace()

    while !self.cursor.consume(")") {
      guard let argumentName = self.parseIdentifier() else { return nil }
      self.cursor.skipWhitespace()
      guard self.cursor.consume("=") else { return nil }
      self.cursor.skipWhitespace()
      guard let value = self.parseValue() else { return nil }
      arguments[argumentName] = value
      self.cursor.skipWhitespace()
      if self.cursor.consume(")") { break }
      guard self.cursor.consume(",") else { return nil }
      self.cursor.skipWhitespace()
    }

    self.cursor.skipWhitespace()
    guard self.cursor.isAtEnd else { return nil }
    return EdgeRawToolCall(name: name, arguments: .object(arguments))
  }

  mutating func parseValue() -> EdgeToolsValue? {
    self.cursor.skipWhitespace()
    guard let character = self.cursor.current else { return nil }
    if character == "'" || character == "\"" {
      return self.parseString(quote: character).map(EdgeToolsValue.string)
    }
    if character == "[" { return self.parseArray() }
    if character == "{" { return self.parseObject() }
    if character == "-" || character == "+" || character.isNumber {
      return self.parseNumber()
    }
    guard let identifier = self.parseIdentifier() else { return nil }
    return parseToolCallBooleanOrNull(identifier)
  }

  private mutating func parseObject() -> EdgeToolsValue? {
    guard self.cursor.consume("{") else { return nil }
    self.cursor.skipWhitespace()
    return self.parseObjectBody().map(EdgeToolsValue.object)
  }

  mutating func parseObjectKey() -> String? {
    guard let quote = self.cursor.current, quote == "'" || quote == "\"" else { return nil }
    guard let key = self.parseString(quote: quote) else { return nil }
    self.cursor.skipWhitespace()
    guard self.cursor.consume(":") else { return nil }
    return key
  }

  private mutating func parseString(quote: Character) -> String? {
    guard self.cursor.consume(character: quote) else { return nil }
    var result = ""
    while let character = self.cursor.current {
      self.cursor.advance()
      if character == quote { return result }
      if character == "\\" {
        guard self.appendEscape(to: &result) else { return nil }
      } else {
        result.append(character)
      }
    }
    return nil
  }

  private mutating func appendEscape(to result: inout String) -> Bool {
    guard let escaped = self.cursor.current else { return false }
    self.cursor.advance()
    if escaped == "u" {
      guard let scalar = self.parseUnicodeEscape() else { return false }
      result.unicodeScalars.append(scalar)
    } else if let character = Self.escapedCharacters[escaped] {
      result.append(character)
    } else {
      result.append("\\")
      result.append(escaped)
    }
    return true
  }

  private mutating func parseUnicodeEscape() -> Unicode.Scalar? {
    guard let first = self.parseFourHexDigits() else { return nil }
    guard (0xD800...0xDBFF).contains(first) else { return Unicode.Scalar(first) }
    guard self.cursor.consume("\\"), self.cursor.consume("u") else { return nil }
    guard let second = self.parseFourHexDigits(), (0xDC00...0xDFFF).contains(second) else {
      return nil
    }
    let value = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
    return Unicode.Scalar(value)
  }

  private mutating func parseFourHexDigits() -> UInt32? {
    var value: UInt32 = 0
    for _ in 0..<4 {
      guard let character = self.cursor.current, let digit = character.hexDigitValue else {
        return nil
      }
      value = value * 16 + UInt32(digit)
      self.cursor.advance()
    }
    return value
  }

  private mutating func parseNumber() -> EdgeToolsValue? {
    let source = self.cursor.read {
      $0.isNumber || ["-", "+", ".", "e", "E"].contains($0)
    }
    if let integer = Int(source) { return .integer(integer) }
    return Double(source).map(EdgeToolsValue.number)
  }

  private mutating func parseIdentifier() -> String? {
    self.cursor.skipWhitespace()
    let identifier = self.cursor.read {
      !$0.isWhitespace && !["=", "(", ")", "[", "]", "{", "}", ":", ","].contains($0)
    }
    return identifier.isEmpty ? nil : identifier
  }
}

// MARK: - LFM2 Grammar

#if XGrammar
  extension XGRGrammar {
    public static func lfm2(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.lfm2Python(tools: tools, range: range)
    }

    public static func lfm2P5(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.lfm2Python(tools: tools, range: range)
    }

    public static func lfm2Python(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      let calls = try Self.toolCalls(tools: Array(tools), separator: ",", range: range) {
        try Self.pythonCall($0)
      }
      let prefix = try XGRGrammar.literal("<|tool_call_start|>[")
      let withCalls = try prefix.concatenate(calls)
      let grammar = try withCalls.concatenate(
        XGRGrammar.literal("]<|tool_call_end|>")
      )
      var document = try XGREBNFDocument(grammar.ebnf)
      document.removeDuplicateRules()
      return try XGRGrammar.ebnf(document.source)
    }

    private static func pythonCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let jsonArguments = Self.strictJSONArguments(for: tool)
      var document = try XGREBNFDocument(jsonArguments.ebnf)
      try document.mapLiterals { ruleName, value, suffix in
        switch value {
        case "true": return "True"
        case "false": return "False"
        case "null": return "None"
        default: break
        }

        guard Self.isLFMTopLevelArgumentRule(ruleName) else { return value }
        if value == "{" || value == "}" { return "" }
        if value == ":" { return "=" }
        if value.count >= 2, value.first == "\"", value.last == "\"",
          #"":""#.firstRange(in: suffix) != nil
        {
          return String(value.dropFirst().dropLast())
        }
        return value
      }

      let arguments = try XGRGrammar.ebnf(document.source)
      let prefix = try XGRGrammar.literal("\(tool.name)(")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar.literal(")"))
    }

    private static func isLFMTopLevelArgumentRule(_ name: String) -> Bool {
      guard name != "root" else { return true }
      guard name.starts(with: "root_") else { return false }
      var digits = name.dropFirst("root_".count)
      if digits.starts(with: "part_") { digits = digits.dropFirst("part_".count) }
      return !digits.isEmpty && digits.allSatisfy { $0.isASCII && $0.isNumber }
    }
  }
#endif
