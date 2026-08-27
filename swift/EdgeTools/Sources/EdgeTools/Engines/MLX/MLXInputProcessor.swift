#if MLX && canImport(MLX)
  import EdgeToolsCore
  import EdgeToolsTokenizers
  import MLXLMCommon

  // MARK: - MLXInputProcessor

  actor MLXInputProcessor<Profile: MLXModelProfile>
  where Profile.Prompt == EdgeToolsTranscript {
    private let processor: (any UserInputProcessor)?
    private let tokenizer: any EdgeToolsTokenizer

    init(processor: (any UserInputProcessor)?, tokenizer: any EdgeToolsTokenizer) {
      self.processor = processor
      self.tokenizer = tokenizer
    }

    func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      kind: EdgeToolsLLMInputKind
    ) async throws -> sending LMInput {
      switch kind {
      case .generation:
        return try await Profile.input(
          prompt: prompt,
          reasoningEffort: reasoningEffort,
          tools: tools,
          tokenizer: self.tokenizer,
          processor: self.processor
        )
      case .prefill:
        return try await Profile.prefillInput(
          prompt: prompt,
          reasoningEffort: reasoningEffort,
          tools: tools,
          tokenizer: self.tokenizer,
          processor: self.processor
        )
      }
    }

    func tokenIds(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition]
    ) async throws -> [EdgeToolsToken.ID] {
      let input = try await self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        kind: .generation
      )
      return input.text.tokens.asArray(EdgeToolsToken.ID.self)
    }
  }
#endif
