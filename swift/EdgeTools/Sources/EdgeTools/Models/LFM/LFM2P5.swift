#if XGrammar
  import EdgeToolsXGrammar

  // MARK: - LFM2P5 Model

  public struct LFM2P5Profile: EdgeToolsModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = LFM2P5GenerationParser
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
        try .lfm2P5(tools: tools, range: $0)
      }
    }
  }
#endif

#if MLX && canImport(MLX)
  extension LFM2P5Profile: MLXLLMModelProfile {}

  public typealias LFM2P5MLXModelEngine = MLXEngine<LFM2P5Profile>
#endif

#if Llama && canImport(CLlama)
  extension LFM2P5Profile: LlamaModelProfile {}

  public typealias LFM2P5LlamaModelEngine = LlamaEngine<LFM2P5Profile>
#endif
