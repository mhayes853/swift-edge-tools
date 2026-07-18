#if MLX && canImport(MLX)
  import MLXLMCommon

  // MARK: - EdgeToolsMLXModelConfiguration

  public protocol EdgeToolsMLXModelConfiguration: Sendable {
    associatedtype ModelConfiguration: Decodable
    associatedtype Prompt: Sendable
    associatedtype ToolCallParser: EdgeToolCallParser
    associatedtype LanguageModel: MLXLMCommon.LanguageModel

    static func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGrammarGrammar

    static func grammarCompiler(
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> XGrammarCompiler

    static func languageModel(configuration: ModelConfiguration) -> sending LanguageModel

    static func tokenize(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> sending LMInput
  }

  // MARK: - EdgeToolsPrefillableMLXModelConfiguration

  public protocol EdgeToolsPrefillableMLXModelConfiguration:
    EdgeToolsMLXModelConfiguration {}
#endif
