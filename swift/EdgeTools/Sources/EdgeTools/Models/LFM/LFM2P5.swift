#if MLX && canImport(MLX)
  import EdgeToolsXGrammar

  public struct LFM2P5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: MLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .lfm2P5(tools: tools, range: $0)
      }
    }
  }

  public typealias LFM2P5MLXModelEngine = MLXEngine<LFM2P5MLXProfile>
#endif

#if Llama && canImport(CLlama)
  import EdgeToolsXGrammar

  public struct LFM2P5LlamaProfile: LlamaModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: LlamaGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .lfm2P5(tools: tools, range: $0)
      }
    }
  }

  public typealias LFM2P5LlamaModelEngine = LlamaEngine<LFM2P5LlamaProfile>
#endif
