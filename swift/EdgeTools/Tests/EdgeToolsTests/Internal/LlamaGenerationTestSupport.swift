#if Llama && XGrammar && canImport(CLlama)
  import EdgeTools

  func completeWeatherTurn<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> EdgeToolsTranscript
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    let turn = try splitUserMessage(from: .weatherTest)
    let context = engine.context(
      EdgeToolsTranscriptContextParameters(transcript: turn.transcript)
    )
    return try await completeToolTurn(
      in: context,
      tool: .weatherTest,
      toolResponse: .weatherTestResponse,
      generatingToolCall: {
        try engine.generate(
          prompt: turn.userMessage,
          tools: [.weatherTest],
          parameters: DefaultLlamaGenerateParameters(
            sampling: .greedy,
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: 256
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: {
        try engine.generate(
          tools: [],
          parameters: DefaultLlamaGenerateParameters(maxTokens: 64),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      }
    )
    .transcript
  }

  func generateReasoning<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> EdgeToolsEngineGeneration
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    try await reasoningGeneration(
      from: try engine.generate(
        prompt: .user(.reasoningTest),
        tools: [],
        parameters: DefaultLlamaGenerateParameters(maxTokens: 512),
        context: engine.context(
          EdgeToolsTranscriptContextParameters(reasoningEffort: .high)
        ),
        channel: EdgeToolsGenerationChannel()
      )
    )
  }
#endif
