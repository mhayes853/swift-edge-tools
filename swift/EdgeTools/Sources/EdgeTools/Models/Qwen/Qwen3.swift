#if XGrammar
  import EdgeToolsCore
  import EdgeToolsXGrammar

  // MARK: - Qwen3 Model

  public struct Qwen3Profile: EdgeToolsModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = Qwen3GenerationParser
    public typealias GrammarEngine = XGrammarEngine
    public typealias Constraint = XGRGenerationConstraint

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
        try XGRGrammar.qwen3(tools: tools, range: $0)
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

    public static func defaultSampling(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort
    ) -> EdgeToolsFusedSamplingParameters? {
      reasoningEffort.isEnabled
        ? EdgeToolsFusedSamplingParameters(temperature: 0.6, topK: 20, topP: 0.95)
        : EdgeToolsFusedSamplingParameters(temperature: 0.7, topK: 20, topP: 0.8)
    }
  }
#endif

#if MLX && canImport(MLX)
  extension Qwen3Profile: MLXLLMModelProfile {}

  public typealias Qwen3MLXModelEngine = MLXEngine<Qwen3Profile>
#endif

#if Llama && canImport(CLlama)
  extension Qwen3Profile: LlamaModelProfile {}

  public typealias Qwen3LlamaModelEngine = LlamaEngine<Qwen3Profile>
#endif

// MARK: - Qwen3 Tool Call Parsing

public struct Qwen3GenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<tool_call>",
    toolCloser: "</tool_call>",
    reasoningOpener: "<think>",
    reasoningCloser: "</think>",
    parseToolCalls: qwenJSONToolCalls(in:)
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}

private func qwenJSONToolCalls(in source: String) -> [EdgeRawToolCall] {
  var block = IncrementalToolCallBlock(opener: "<tool_call>", closer: "</tool_call>")
  block.append(EdgeToolsToken(id: -1, stringValue: source))
  var calls = [EdgeRawToolCall]()
  while let payload = block.nextPayload(respectingJSONStringBoundaries: true) {
    if let value = try? EdgeToolsValue(json: payload), let call = EdgeRawToolCall(jsonValue: value)
    {
      calls.append(call)
    }
  }
  return calls
}

// MARK: - Qwen3 Grammar

#if XGrammar
  extension XGRGrammar {
    static func qwenReasoning() throws -> XGRGrammar {
      let opener = try Self.literal("<think>")
      let thought = try opener.concatenate(.universal)
      return try thought.concatenate(Self.literal("</think>"))
    }

    public static func qwenJSON(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.qwenJSONCall($0)
      }
    }

    public static func qwen3(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.qwenJSON(tools: tools, range: range)
    }

    private static func qwenJSONCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let arguments = Self.strictJSONArguments(for: tool)
      let encodedName = EdgeToolsValue.string(tool.name).orderedJSONString()
      let prefix = try XGRGrammar.literal("<tool_call>{\"name\":\(encodedName),\"arguments\":")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar.literal("}</tool_call>"))
    }
  }
#endif
