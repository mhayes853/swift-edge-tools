#if MLX && XGrammar && canImport(MLX)
  import EdgeToolsXGrammar

  public struct LFM2P5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .lfm2P5(tools: tools, range: $0)
      }
    }
  }

  public typealias LFM2P5MLXModelEngine = MLXEngine<LFM2P5MLXProfile>
#endif
