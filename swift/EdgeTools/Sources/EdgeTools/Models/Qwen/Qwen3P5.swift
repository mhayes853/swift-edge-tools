import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  // MARK: - Qwen3P5 Model

  public struct Qwen3P5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsConversationalPrompt
    public typealias GenerationParser = Qwen3P5GenerationParser
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
        let toolCalls = try XGRGrammar.qwen3P5(tools: tools, range: range)
        guard prompt.reasoningEffort.isEnabled else { return toolCalls }
        return try XGRGrammar.qwenReasoning().concatenate(toolCalls)
      }
    }

    public static func templateContext(
      prompt: EdgeToolsConversationalPrompt
    ) -> [String: any Sendable]? {
      guard prompt.reasoningEffort != .default else { return nil }
      return ["enable_thinking": prompt.reasoningEffort.isEnabled]
    }

    public static func prepare(
      prompt: inout EdgeToolsConversationalPrompt,
      tools: [EdgeToolDefinition],
      parser: inout Qwen3P5GenerationParser
    ) {
      let prefix = prompt.reasoningEffort.isEnabled ? "<think>\n" : "<think>\n\n</think>\n\n"
      _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
    }

    public static func defaultSampling(
      prompt: EdgeToolsConversationalPrompt,
      parameters: DefaultMLXGenerateParameters
    ) -> EdgeToolsFusedSamplingParameters? {
      prompt.reasoningEffort.isEnabled
        ? EdgeToolsFusedSamplingParameters(
          temperature: 1,
          topK: 20,
          topP: 0.95,
          presencePenalty: 1.5
        )
        : EdgeToolsFusedSamplingParameters(temperature: 1, topK: 20, presencePenalty: 2)
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
    public typealias Prompt = EdgeToolsConversationalPrompt
    public typealias GenerationParser = Qwen3P5GenerationParser
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
        let toolCalls = try XGRGrammar.qwen3P5(tools: tools, range: range)
        guard prompt.reasoningEffort.isEnabled else { return toolCalls }
        return try XGRGrammar.qwenReasoning().concatenate(toolCalls)
      }
    }

    public static func templateContext(
      prompt: EdgeToolsConversationalPrompt
    ) -> [String: any Sendable]? {
      guard prompt.reasoningEffort != .default else { return nil }
      return ["enable_thinking": prompt.reasoningEffort.isEnabled]
    }

    public static func prepare(
      prompt: inout EdgeToolsConversationalPrompt,
      tools: [EdgeToolDefinition],
      parser: inout Qwen3P5GenerationParser
    ) {
      let prefix = prompt.reasoningEffort.isEnabled ? "<think>\n" : "<think>\n\n</think>\n\n"
      _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
    }

    public static func defaultSampling(
      prompt: EdgeToolsConversationalPrompt,
      parameters: DefaultMLXGenerateParameters
    ) -> EdgeToolsFusedSamplingParameters? {
      prompt.reasoningEffort.isEnabled
        ? EdgeToolsFusedSamplingParameters(temperature: 0.6, topK: 20, topP: 0.95)
        : EdgeToolsFusedSamplingParameters(
          temperature: 0.7,
          topK: 20,
          topP: 0.8,
          presencePenalty: 1.5
        )
    }

    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsConversationalPrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await prompt.mlxVLMInput(
        tools: tools,
        processor: processor,
        additionalContext: Self.templateContext(prompt: prompt)
      ) { message in
        switch message {
        case .user(let message):
          var content: [MLXLMCommon.Message] = message.images.map { _ in ["type": "image"] }
          content.append(contentsOf: message.videos.map { _ in ["type": "video"] })
          content.append(["type": "text", "text": message.content])
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

private func qwenXMLToolCalls(in source: String) -> [EdgeRawToolCall] {
  var block = IncrementalToolCallBlock(opener: "<tool_call>", closer: "</tool_call>")
  block.append(EdgeToolsToken(id: -1, stringValue: source))
  var calls = [EdgeRawToolCall]()
  while let payloadData = block.nextPayload(
    outsideRegionOpenedBy: QwenXMLToolCalls.parameterOpener,
    closedBy: QwenXMLToolCalls.parameterCloser,
    orAbortedBy: QwenXMLToolCalls.functionCloser
  ) {
    let payload = String(decoding: payloadData, as: UTF8.self)
    if let call = QwenXMLToolCalls.parse(payload) { calls.append(call) }
  }
  return calls
}

private enum QwenXMLToolCalls {
  static let parameterOpener = Array("<parameter=".utf8)
  static let parameterCloser = Array("</parameter>".utf8)
  static let functionCloser = Array("</function>".utf8)

  static func parse(_ payload: String) -> EdgeRawToolCall? {
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
      guard let parameterEnd = "</parameter>".firstRange(in: body[valueStart...]) else {
        return nil
      }
      let name = body[parameterStart.upperBound..<nameEnd].trimmingWhitespace
      guard !name.isEmpty, arguments[name] == nil else { return nil }
      let source = body[valueStart..<parameterEnd.lowerBound].trimmingWhitespace
      arguments[name] = Self.parseValue(source)
      searchStart = parameterEnd.upperBound
    }
    return arguments
  }

  private static func parseValue(_ source: String) -> EdgeToolsValue {
    if let value = parseToolCallBooleanOrNull(source) { return value }
    return (try? EdgeToolsValue(json: source)) ?? .string(source)
  }
}

public struct Qwen3P5GenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<tool_call>",
    toolCloser: "</tool_call>",
    reasoningOpener: "<think>",
    reasoningCloser: "</think>",
    ignoredToolRegions: [
      DelimitedGenerationParser.IgnoredToolRegion(
        opener: "<parameter=",
        closer: "</parameter>"
      )
    ],
    parseToolCalls: qwenXMLToolCalls(in:)
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}

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
