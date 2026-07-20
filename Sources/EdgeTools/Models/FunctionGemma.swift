#if MLX && Transformers && canImport(MLX)
  #if canImport(FoundationEssentials)
    import FoundationEssentials
  #else
    import Foundation
  #endif
  import MLXLLM

  // MARK: - FunctionGemma Model

  extension MLXLLM.Gemma3TextModel: EdgeToolsLanguageModel {
    public typealias ModelConfiguration = MLXLLM.Gemma3TextConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = FunctionGemmaToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.functionGemma(tools: tools, range: range)
    }
  }

  public typealias FunctionGemmaMLXEngine = EdgeToolsMLXEngine<MLXLLM.Gemma3TextModel>

  extension EdgeToolsMLXEngine where Model == MLXLLM.Gemma3TextModel {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: MLXLLM.Gemma3TextModel.init)
    }
  }
#endif

// MARK: - FunctionGemmaToolCallParser

public struct FunctionGemmaToolCallParser: EdgeToolCallParser, Sendable {
  private static let stringMarker = Array("<escape>".utf8)

  private var block = IncrementalToolCallBlock(
    opener: "<start_function_call>",
    closer: "<end_function_call>"
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.block.append(token)
    while let payloadData = self.block.nextPayload(outside: Self.stringMarker) {
      let payload = String(decoding: payloadData, as: UTF8.self)
      var reader = FunctionGemmaCallReader(source: payload)
      if let call = reader.parse() { return call }
    }
    return nil
  }
}

// MARK: - FunctionGemmaCallReader

private struct FunctionGemmaCallReader: ToolCallValueReader {
  private static let stringMarker = "<escape>"

  var cursor: ToolCallStringCursor

  init(source: String) {
    self.cursor = ToolCallStringCursor(source)
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
      let key = self.cursor.read(until: ":")?.trimmingWhitespace,
      !key.isEmpty,
      self.cursor.consume(":")
    else { return nil }
    return key
  }

  mutating func parseValue() -> EdgeToolsValue? {
    self.cursor.skipWhitespace()
    if self.cursor.remainder.hasPrefix(Self.stringMarker) {
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
    guard self.cursor.consume(Self.stringMarker) else { return nil }
    guard let value = self.cursor.read(until: Self.stringMarker) else { return nil }
    guard self.cursor.consume(Self.stringMarker) else { return nil }
    return (try? decodeEdgeToolsJSON(Array(value.utf8))) ?? .string(value)
  }

  private mutating func readBareValue() -> String {
    self.cursor.read { ![",", "}", "]"].contains($0) }.trimmingWhitespace
  }

  private static func parseBareValue(_ token: String) -> EdgeToolsValue? {
    return switch token {
    case "true", "True": true
    case "false", "False": false
    case "null", "None": .null
    default:
      if token.isEmpty {
        nil
      } else if let integer = Int(token) {
        .integer(integer)
      } else if let number = Double(token) {
        .number(number)
      } else {
        .string(token)
      }
    }
  }
}

// MARK: - FunctionGemma Grammar

#if XGrammar
  extension XGRGrammar {
    public static func functionGemma(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.functionGemmaCall($0)
      }
    }

    private static func functionGemmaCall(
      _ tool: EdgeToolDefinition
    ) throws -> XGRGrammar {
      let xmlArguments = Self.qwenXMLArguments(for: tool)
      var document = try XGREBNFDocument(xmlArguments.ebnf)
      try document.mapLiterals { ruleName, value, suffix in
        if value == "</parameter>" {
          if ruleName == "xml_string" { return "<escape>" }
          return suffix.hasToolCallContinuationReference ? "<escape>," : "<escape>"
        }
        if value.hasPrefix("<parameter="), value.hasSuffix(">") {
          let name = value.dropFirst("<parameter=".count).dropLast()
          return "\(name):<escape>"
        }
        if value == "<parameter=" { return "" }
        if value == ">", ruleName.hasPrefix("xml_object") {
          return ":<escape>"
        }
        return value
      }

      let arguments = try XGRGrammar(ebnf: document.source)
      let prefix = try XGRGrammar(
        literal: "<start_function_call>call:\(tool.name){"
      )
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGRGrammar(literal: "}<end_function_call>")
      )
    }

  }
#endif
