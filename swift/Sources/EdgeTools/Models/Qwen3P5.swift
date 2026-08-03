import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && Transformers && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - Qwen3P5 Model

  extension Qwen35Model: EdgeToolsMLXModel {
    public typealias ModelConfiguration = Qwen35Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Qwen3P5ToolCallParser

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .qwen3P5(tools: tools, range: range)
    }
  }

  public typealias Qwen35MLXModelEngine = EdgeToolsMLXEngine<Qwen35Model>
  public typealias Qwen3P5MLXModelEngine = EdgeToolsMLXEngine<Qwen35Model>

  extension EdgeToolsMLXEngine where Model == Qwen35Model {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: Qwen35Model.init)
    }
  }
#endif

// MARK: - Qwen3P5 Tool Call Parsing

public struct QwenXMLToolCallParser: EdgeToolCallParser, Sendable {
  private static let parameterOpener = Array("<parameter=".utf8)
  private static let parameterCloser = Array("</parameter>".utf8)
  private static let functionCloser = Array("</function>".utf8)

  private var block = IncrementalToolCallBlock(opener: "<tool_call>", closer: "</tool_call>")

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.block.append(token)
    while let payloadData = self.nextBlockPayload() {
      let payload = String(decoding: payloadData, as: UTF8.self)
      if let call = Self.parse(payload) {
        return call
      }
    }
    return nil
  }

  private mutating func nextBlockPayload() -> [UInt8]? {
    self.block.nextPayload(
      outsideRegionOpenedBy: Self.parameterOpener,
      closedBy: Self.parameterCloser,
      orAbortedBy: Self.functionCloser
    )
  }

  private static func parse(_ payload: String) -> EdgeRawToolCall? {
    guard let functionStart = payload[...].firstRange(of: "<function=") else { return nil }
    guard let nameEnd = payload[functionStart.upperBound...].firstIndex(of: ">") else { return nil }
    guard let functionEnd = payload[nameEnd...].firstRange(of: "</function>") else { return nil }

    let name = payload[functionStart.upperBound..<nameEnd].trimmingWhitespace
    guard !name.isEmpty else { return nil }

    let bodyStart = payload.index(after: nameEnd)
    let body = payload[bodyStart..<functionEnd.lowerBound]
    guard let arguments = Self.parseParameters(body) else { return nil }
    return EdgeRawToolCall(name: name, arguments: .object(arguments))
  }

  private static func parseParameters(
    _ body: Substring
  ) -> OrderedDictionary<String, EdgeToolsValue>? {
    var arguments = OrderedDictionary<String, EdgeToolsValue>()
    var searchStart = body.startIndex

    while let parameterStart = body[searchStart...].firstRange(of: "<parameter=") {
      guard let nameEnd = body[parameterStart.upperBound...].firstIndex(of: ">") else { return nil }
      let valueStart = body.index(after: nameEnd)
      guard
        let parameterEnd = body[valueStart...].firstRange(of: "</parameter>")
      else { return nil }

      let name = body[parameterStart.upperBound..<nameEnd].trimmingWhitespace
      guard !name.isEmpty else { return nil }
      let source = body[valueStart..<parameterEnd.lowerBound].trimmingWhitespace
      arguments[name] = Self.parseParameterValue(source)
      searchStart = parameterEnd.upperBound
    }
    return arguments
  }

  private static func parseParameterValue(_ source: String) -> EdgeToolsValue {
    switch source {
    case "True": return true
    case "False": return false
    case "None": return nil
    default:
      return (try? EdgeToolsJSONDecoder().decode(EdgeToolsValue.self, from: Array(source.utf8)))
        ?? .string(source)
    }
  }
}

public typealias Qwen3P5ToolCallParser = QwenXMLToolCallParser
public typealias Qwen3P6ToolCallParser = QwenXMLToolCallParser

// MARK: - Qwen3P5 Grammar

#if XGrammar
  extension XGRGrammar {
    public static func qwenXML(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.qwenXMLCall($0)
      }
    }

    public static func qwen3P5(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.qwenXML(tools: tools, range: range)
    }

    public static func qwen3P6(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.qwenXML(tools: tools, range: range)
    }

    private static func qwenXMLCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let arguments = Self.qwenXMLArguments(for: tool)
      let prefix = try XGRGrammar.literal("<tool_call><function=\(tool.name)>")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGRGrammar.literal("</function></tool_call>")
      )
    }
  }
#endif
