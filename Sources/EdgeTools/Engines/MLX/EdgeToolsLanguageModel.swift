#if MLX && canImport(MLX)
  import MLXLMCommon

  // MARK: - EdgeToolsLanguageModel

  public protocol EdgeToolsLanguageModel: LanguageModel {
    associatedtype ModelConfiguration: Decodable
    associatedtype Prompt: Sendable
    associatedtype ToolCallParser: EdgeToolCallParser

    init(configuration: ModelConfiguration)

    func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGrammarGrammar

    func grammarCompiler(
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> XGrammarCompiler

    func tokenize(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> sending LMInput
  }

#endif
