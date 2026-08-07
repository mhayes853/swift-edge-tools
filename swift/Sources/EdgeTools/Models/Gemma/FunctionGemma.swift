#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && Transformers && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - FunctionGemma Model

  public struct FunctionGemmaMLXModel: MLXModel {
    public typealias ModelConfiguration = MLXLLM.Gemma3TextConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = FunctionGemmaToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public let languageModel: MLXLLM.Gemma3TextModel

    public init(configuration: MLXLLM.Gemma3TextConfiguration) {
      self.languageModel = MLXLLM.Gemma3TextModel(configuration)
    }

    public var vocabularySize: Int { self.languageModel.vocabularySize }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .functionGemma(tools: tools, range: range)
    }
  }

  public typealias FunctionGemmaMLXModelEngine = MLXEngine<FunctionGemmaMLXModel>
#endif
