#if MLX && canImport(MLX)
  import MLXLMCommon

  public protocol EdgeToolsLanguageModel: LanguageModel {
    associatedtype ModelConfiguration: Decodable
    associatedtype Prompt: Sendable
    associatedtype ToolCallParser: EdgeToolCallParser

    var vocabularySize: Int { get }

    func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar

    func grammarCompiler(
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> XGRCompiler

    func process(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> sending LMInput
  }

  extension EdgeToolsLanguageModel {
    public func grammarCompiler(
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> XGRCompiler {
      #if Transformers
        if let tokenizer = tokenizer as? EdgeToolsPreTrainedTokenizer {
          let vocabulary = tokenizer.convertIdsToTokens(Array(0..<self.vocabularySize))
            .map { $0 ?? "" }
          let tokenizerInfo = try XGRTokenizerInfo.huggingFace(
            encodedVocabulary: vocabulary,
            backendJSON: tokenizer.backendJSON,
            stopTokenIDs: tokenizer.eosTokenId.map { [$0] } ?? []
          )
          return try XGRCompiler(tokenizerInfo: tokenizerInfo)
        }
      #endif

      guard let tokenizer = tokenizer as? NeedleSPTokenizer else {
        throw EdgeToolsMLXEngineError.unsupportedTokenizer
      }
      let tokenizerInfo = try XGRTokenizerInfo.needle(tokenizer: tokenizer)
      return try XGRCompiler(tokenizerInfo: tokenizerInfo)
    }
  }
#endif
