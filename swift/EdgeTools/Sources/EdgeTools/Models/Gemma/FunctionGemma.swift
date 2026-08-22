#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  import EdgeToolsCore

  public struct FunctionGemmaMLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = FunctionGemmaGenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: MLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      ) {
        try .functionGemma(tools: tools, range: $0)
      }
    }
  }

  public typealias FunctionGemmaMLXModelEngine = MLXEngine<FunctionGemmaMLXProfile>
#endif

#if Llama && XGrammar && canImport(CLlama)
  import EdgeToolsCore

  public struct FunctionGemmaLlamaProfile: LlamaModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = FunctionGemmaGenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: LlamaGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      ) {
        try .functionGemma(tools: tools, range: $0)
      }
    }
  }

  public typealias FunctionGemmaLlamaModelEngine = LlamaEngine<FunctionGemmaLlamaProfile>
#endif

public struct FunctionGemmaGenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<start_function_call>",
    toolCloser: "<end_function_call>",
    parseToolCalls: { GemmaToolCalls.parse($0, format: .functionGemma) }
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}
