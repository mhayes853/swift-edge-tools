#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  import EdgeToolsCore

  // MARK: - Qwen3 Model

  public struct Qwen3MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = Qwen3GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      ) { range in
        let toolCalls = try XGRGrammar.qwen3(tools: tools, range: range)
        guard prompt.reasoningEffort.isEnabled else { return toolCalls }
        return try XGRGrammar.qwenReasoning().concatenate(toolCalls)
      }
    }

    public static func templateContext(prompt: EdgeToolsTranscript) -> [String: EdgeToolsValue]? {
      guard prompt.reasoningEffort != .default else { return nil }
      return ["enable_thinking": .boolean(prompt.reasoningEffort.isEnabled)]
    }

    public static func defaultSampling(
      prompt: EdgeToolsTranscript,
      parameters: DefaultMLXGenerateParameters
    ) -> EdgeToolsFusedSamplingParameters? {
      prompt.reasoningEffort.isEnabled
        ? EdgeToolsFusedSamplingParameters(temperature: 0.6, topK: 20, topP: 0.95)
        : EdgeToolsFusedSamplingParameters(temperature: 0.7, topK: 20, topP: 0.8)
    }
  }

  public typealias Qwen3MLXModelEngine = MLXEngine<Qwen3MLXProfile>
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
