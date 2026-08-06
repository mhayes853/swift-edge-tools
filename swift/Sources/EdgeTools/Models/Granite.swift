#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && Transformers && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - Granite Model

  extension GraniteModel: MLXModel {
    public typealias ModelConfiguration = GraniteConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = GraniteToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .granite(tools: tools, range: range)
    }
  }

  public typealias GraniteMLXModelEngine = MLXEngine<GraniteModel>

  extension GraniteMLXModelEngine {
    public init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: GraniteModel.init)
    }
  }

  // MARK: - GraniteMoeHybrid Model

  extension GraniteMoeHybridModel: MLXModel {
    public typealias ModelConfiguration = GraniteMoeHybridConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = GraniteToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .granite(tools: tools, range: range)
    }
  }

  public typealias GraniteMoeHybridMLXModelEngine = MLXEngine<GraniteMoeHybridModel>

  extension GraniteMoeHybridMLXModelEngine {
    public init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: GraniteMoeHybridModel.init)
    }
  }
#endif

// MARK: - Granite Tool Call Parsing

public struct GraniteToolCallParser: EdgeToolCallParser, Sendable {
  private var parser = QwenJSONToolCallParser()

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
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
      try Self.qwenJSON(tools: tools, range: range)
    }
  }
#endif
