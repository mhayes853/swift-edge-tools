#if XGrammar
  import EdgeToolsCore
  import EdgeToolsXGrammar

  // MARK: - FunctionGemma Model

  public struct FunctionGemmaProfile: EdgeToolsModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = FunctionGemmaGenerationParser
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
        try .functionGemma(tools: tools, range: $0)
      }
    }
  }
#endif

#if MLX && canImport(MLX)
  extension FunctionGemmaProfile: MLXLLMModelProfile {}

  public typealias FunctionGemmaMLXModelEngine = MLXEngine<FunctionGemmaProfile>
#endif

#if Llama && canImport(CLlama)
  extension FunctionGemmaProfile: LlamaModelProfile {}

  public typealias FunctionGemmaLlamaModelEngine = LlamaEngine<FunctionGemmaProfile>
#endif

// MARK: - FunctionGemma Tool Call Parsing

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
