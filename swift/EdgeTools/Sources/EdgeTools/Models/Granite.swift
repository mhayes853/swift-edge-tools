#if XGrammar
  import EdgeToolsCore
  import EdgeToolsXGrammar

  // MARK: - Granite Model

  public struct GraniteProfile: EdgeToolsModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = GraniteGenerationParser
    public typealias GrammarEngine = XGrammarEngine
    public typealias Constraint = XGRGenerationConstraint

    public static func grammar(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      constraint: XGRGenerationConstraint,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(
        tools: tools,
        constraint: constraint,
        grammarEngine: grammarEngine
      ) {
        try .granite(tools: tools, range: $0)
      }
    }
  }

  // MARK: - GraniteMoeHybrid Model

  public typealias GraniteMoeHybridProfile = GraniteProfile
#endif

#if MLX && canImport(MLX)
  extension GraniteProfile: MLXLLMModelProfile {}

  public typealias GraniteMLXModelEngine = MLXEngine<GraniteProfile>
  public typealias GraniteMoeHybridMLXModelEngine = MLXEngine<GraniteMoeHybridProfile>
#endif

#if Llama && canImport(CLlama)
  extension GraniteProfile: LlamaModelProfile {}

  public typealias GraniteLlamaModelEngine = LlamaEngine<GraniteProfile>
#endif

// MARK: - Granite Tool Call Parsing

public struct GraniteGenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<tool_call>",
    toolCloser: "</tool_call>",
    parseToolCalls: graniteToolCalls(in:)
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}

private func graniteToolCalls(in source: String) -> [EdgeRawToolCall] {
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

// MARK: - Granite Grammar

#if XGrammar
  extension XGRGrammar {
    public static func granite(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try .qwenJSON(tools: tools, range: range)
    }
  }
#endif
