#if MLX && XGrammar && canImport(MLX)
  import EdgeToolsXGrammar

  public struct LFM2P5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsConversationalPrompt
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func grammar(
      prompt _: EdgeToolsConversationalPrompt,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, context: context) {
        try XGRGrammar.lfm2P5(tools: tools, range: $0)
      }
    }
  }

  public typealias LFM2P5MLXModelEngine = MLXEngine<LFM2P5MLXProfile>
#endif
