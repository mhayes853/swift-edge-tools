import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  // MARK: - Qwen3P5 Model

  public struct Qwen3P5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Qwen3P5ToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .qwen3P5(tools: tools, range: range)
    }
  }

  public typealias Qwen3P5MLXModelEngine = MLXEngine<Qwen3P5MLXProfile>
#endif

#if MLX && XGrammar && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import Foundation
  import MLXLMCommon
  import MLXVLM

  // MARK: - Qwen3P5 VLM Model

  public struct Qwen3P5VLMLXProfile: MLXVLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Qwen3P5ToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .qwen3P5(tools: tools, range: range)
    }

    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      tokenizer _: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await prompt.mlxVLMInput(tools: tools, processor: processor) { message in
        switch message {
        case .user(let text, let images, let videos, audio: _):
          var content: [MLXLMCommon.Message] = images.map { _ in ["type": "image"] }
          content.append(contentsOf: videos.map { _ in ["type": "video"] })
          content.append(["type": "text", "text": text])
          return ["role": "user", "content": content]
        case .system, .assistant, .tool:
          return try message.mlxMessage()
        }
      }
    }
  }

  public typealias Qwen3P5VLMLXModelEngine = MLXEngine<Qwen3P5VLMLXProfile>
#endif

// MARK: - Qwen3P5 Tool Call Parsing

public struct QwenXMLToolCallParser: EdgeToolCallParser, Sendable {
  private static let parameterOpener = Array("<parameter=".utf8)
  private static let parameterCloser = Array("</parameter>".utf8)
  private static let functionCloser = Array("</function>".utf8)

  private var block = IncrementalToolCallBlock(opener: "<tool_call>", closer: "</tool_call>")

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
    guard let functionStart = "<function=".firstRange(in: payload[...]) else { return nil }
    guard let nameEnd = payload[functionStart.upperBound...].firstIndex(of: ">") else { return nil }
    guard let functionEnd = "</function>".firstRange(in: payload[nameEnd...]) else { return nil }

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

    while let parameterStart = "<parameter=".firstRange(in: body[searchStart...]) {
      guard let nameEnd = body[parameterStart.upperBound...].firstIndex(of: ">") else { return nil }
      let valueStart = body.index(after: nameEnd)
      guard
        let parameterEnd = "</parameter>".firstRange(in: body[valueStart...])
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
    if let value = parseToolCallBooleanOrNull(source) {
      return value
    }
    return (try? EdgeToolsValue(json: source)) ?? .string(source)
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
      let arguments = Self.xmlToolArguments(for: tool)
      let prefix = try XGRGrammar.literal("<tool_call><function=\(tool.name)>")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGRGrammar.literal("</function></tool_call>")
      )
    }
  }
#endif
