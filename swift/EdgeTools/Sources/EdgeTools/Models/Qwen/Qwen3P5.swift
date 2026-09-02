import EdgeToolsCore
import OrderedCollections

#if XGrammar
  import EdgeToolsXGrammar

  // MARK: - Qwen3P5ModelProfile

  /// Shared behaviour for the Qwen3.5 family, whose text and vision variants differ only in their
  /// default sampling and in how the MLX engine builds model input.
  public protocol Qwen3P5ModelProfile: EdgeToolsMultimodalModelProfile
  where
    GenerationParser == Qwen3P5GenerationParser,
    GrammarEngine == XGrammarEngine,
    Constraint == XGRGenerationConstraint
  {}

  extension Qwen3P5ModelProfile {
    public static func grammar(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      constraint: XGRGenerationConstraint,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      let grammar = try Self.constrainedGrammar(
        tools: tools,
        constraint: constraint,
        grammarEngine: grammarEngine
      ) {
        try XGRGrammar.qwen3P5(tools: tools, range: $0)
      }
      guard reasoningEffort.isEnabled else { return grammar }
      return try XGRGrammar.qwenReasoning().concatenate(grammar)
    }

    public static func templateContext(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort
    ) -> [String: EdgeToolsValue]? {
      guard reasoningEffort != .default else { return nil }
      return ["enable_thinking": .boolean(reasoningEffort.isEnabled)]
    }

    public static func prepare(
      prompt: inout EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      parser: inout Qwen3P5GenerationParser
    ) {
      let prefix = reasoningEffort.isEnabled ? "<think>\n" : "<think>\n\n</think>\n\n"
      _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
    }
  }

  // MARK: - Qwen3P5 Model

  public struct Qwen3P5Profile: Qwen3P5ModelProfile {
    public static func defaultSampling(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort
    ) -> EdgeToolsFusedSamplingParameters? {
      reasoningEffort.isEnabled
        ? EdgeToolsFusedSamplingParameters(
          temperature: 1,
          topK: 20,
          topP: 0.95,
          presencePenalty: 1.5
        )
        : EdgeToolsFusedSamplingParameters(temperature: 1, topK: 20, presencePenalty: 2)
    }
  }

  // MARK: - Qwen3P5VL Model

  public struct Qwen3P5VLProfile: Qwen3P5ModelProfile {
    public static func defaultSampling(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort
    ) -> EdgeToolsFusedSamplingParameters? {
      reasoningEffort.isEnabled
        ? EdgeToolsFusedSamplingParameters(temperature: 0.6, topK: 20, topP: 0.95)
        : EdgeToolsFusedSamplingParameters(
          temperature: 0.7,
          topK: 20,
          topP: 0.8,
          presencePenalty: 1.5
        )
    }
  }
#endif

#if MLX && canImport(MLX)
  // MARK: - Qwen3P5 MLX Model

  extension Qwen3P5Profile: MLXLLMModelProfile {}

  public typealias Qwen3P5MLXModelEngine = MLXEngine<Qwen3P5Profile>
#endif

#if MLX && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import EdgeToolsTokenizers
  import MLXLMCommon
  import MLXVLM

  // MARK: - Qwen3P5VL MLX Model

  extension Qwen3P5VLProfile: MLXVLMModelProfile {
    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      try await self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        processor: processor,
        addGenerationPrompt: true
      )
    }

    public static nonisolated(nonsending) func prefillInput(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      try await self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        processor: processor,
        addGenerationPrompt: false
      )
    }

    private static nonisolated(nonsending) func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      processor: (any UserInputProcessor)?,
      addGenerationPrompt: Bool
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      var templateContext = Self.templateContext(
        prompt: prompt,
        reasoningEffort: reasoningEffort
      ) ?? [:]
      templateContext["add_generation_prompt"] = .boolean(addGenerationPrompt)
      return try await prompt.mlxVLMInput(
        tools: tools,
        processor: processor,
        additionalContext: templateContext
      ) { message in
        switch message {
        case .user(let message):
          return [
            "role": "user",
            "content": Self.multimodalContent(for: message).map(\.mlxMessage)
          ]
        case .system, .assistant, .tool:
          return try message.mlxMessage()
        }
      }
    }
  }

  public typealias Qwen3P5VLMLXModelEngine = MLXEngine<Qwen3P5VLProfile>
#endif

#if Llama && canImport(CLlama)
  // MARK: - Qwen3P5 Llama Model

  extension Qwen3P5Profile: LlamaModelProfile {}
  extension Qwen3P5VLProfile: LlamaModelProfile {}

  public typealias Qwen3P5LlamaModelEngine = LlamaEngine<Qwen3P5Profile>
  public typealias Qwen3P5VLLlamaModelEngine = LlamaEngine<Qwen3P5VLProfile>
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
    // pi-lens-ignore: optional_data_string_conversion
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
      return try withArguments.concatenate(.literal("</function></tool_call>"))
    }
  }
#endif
