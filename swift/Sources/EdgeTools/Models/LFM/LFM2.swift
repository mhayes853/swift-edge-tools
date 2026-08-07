#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && Transformers && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - LFM2 Model

  public struct LFM2MLXModel: MLXModel {
    public typealias ModelConfiguration = LFM2Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = LFM2ToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public let languageModel: LFM2Model

    public init(configuration: LFM2Configuration) {
      self.languageModel = LFM2Model(configuration)
    }

    public var vocabularySize: Int { self.languageModel.vocabularySize }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .lfm2(tools: tools, range: range)
    }
  }

  public typealias LFM2P5MLXModel = LFM2MLXModel

  public typealias LFM2MLXModelEngine = MLXEngine<LFM2MLXModel>
  public typealias LFM2P5MLXModelEngine = MLXEngine<LFM2P5MLXModel>
#endif
