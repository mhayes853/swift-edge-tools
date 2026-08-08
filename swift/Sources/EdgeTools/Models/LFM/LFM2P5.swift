#if MLX && XGrammar && canImport(MLX)
  import EdgeToolsXGrammar

  public struct LFM2P5MLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = LFM2P5ToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .lfm2P5(tools: tools, range: range)
    }
  }

  public typealias LFM2P5MLXModelEngine = MLXEngine<LFM2P5MLXProfile>
#endif
