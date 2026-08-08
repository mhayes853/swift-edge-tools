#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(MLX)
  // MARK: - Granite Model

  public struct GraniteMLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = GraniteToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .granite(tools: tools, range: range)
    }
  }

  public typealias GraniteMLXModelEngine = MLXEngine<GraniteMLXProfile>

  // MARK: - GraniteMoeHybrid Model

  public struct GraniteMoeHybridMLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = GraniteToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .granite(tools: tools, range: range)
    }
  }

  public typealias GraniteMoeHybridMLXModelEngine = MLXEngine<GraniteMoeHybridMLXProfile>
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
