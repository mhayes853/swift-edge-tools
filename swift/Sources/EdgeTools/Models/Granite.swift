#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && Transformers && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - Granite Model

  public struct GraniteMLXModel: MLXModel {
    public typealias ModelConfiguration = GraniteConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = GraniteToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public let languageModel: GraniteModel

    public init(configuration: GraniteConfiguration) {
      self.languageModel = GraniteModel(configuration)
    }

    public var vocabularySize: Int { self.languageModel.vocabularySize }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .granite(tools: tools, range: range)
    }
  }

  public typealias GraniteMLXModelEngine = MLXEngine<GraniteMLXModel>

  // MARK: - GraniteMoeHybrid Model

  public struct GraniteMoeHybridMLXModel: MLXModel {
    public typealias ModelConfiguration = GraniteMoeHybridConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = GraniteToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public let languageModel: GraniteMoeHybridModel

    public init(configuration: GraniteMoeHybridConfiguration) {
      self.languageModel = GraniteMoeHybridModel(configuration)
    }

    public var vocabularySize: Int { self.languageModel.vocabularySize }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .granite(tools: tools, range: range)
    }
  }

  public typealias GraniteMoeHybridMLXModelEngine = MLXEngine<GraniteMoeHybridMLXModel>
#endif

// MARK: - Granite Tool Call Parsing

public struct GraniteToolCallParser: EdgeToolCallParser, Sendable {
  private var parser = QwenJSONToolCallParser()

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeRawToolCall] {
    self.parser.accept(token: token)
  }
}

// MARK: - Granite Grammar

#if XGrammar
  extension XGRGrammar {
    public static func granite(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try .qwenJSON(tools: tools, range: range)
    }
  }
#endif
