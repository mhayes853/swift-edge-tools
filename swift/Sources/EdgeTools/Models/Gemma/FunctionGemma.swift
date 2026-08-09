#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  public struct FunctionGemmaMLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsConversationalPrompt
    public typealias GenerationParser = FunctionGemmaGenerationParser
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
        try XGRGrammar.functionGemma(tools: tools, range: $0)
      }
    }
  }

  public typealias FunctionGemmaMLXModelEngine = MLXEngine<FunctionGemmaMLXProfile>
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
