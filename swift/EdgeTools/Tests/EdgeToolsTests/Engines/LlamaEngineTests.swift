#if Llama && XGrammar && canImport(CLlama) && !os(WASI)
  import CustomDump
  import EdgeTools
  import EdgeToolsXGrammar
  import Testing

  @Suite(.serialized)
  struct `LlamaEngine tests` {
    @Test
    func `Continuing A Context Only Prefills The Unseen Suffix`() async throws {
      let engine = try await qwen3LlamaEngine()
      let context = engine.context(llamaCachedContextParameters())
      let prefill = try await engine.prefill(context: context)

      let cached = try await singleTokenGeneration(using: engine, context: context)
      let fresh = try await singleTokenGeneration(
        using: engine,
        context: engine.context(llamaCachedContextParameters())
      )

      expectNoDifference(
        cached.prefillMetrics.tokens + prefill.metrics.tokens,
        fresh.prefillMetrics.tokens
      )
    }

    @Test
    func `Forking Leaves The Parent Cache Intact`() async throws {
      let engine = try await qwen3LlamaEngine()
      let context = engine.context(llamaCachedContextParameters())
      _ = try await engine.prefill(context: context)

      let forked = try await singleTokenGeneration(using: engine, context: context.fork())
      let parent = try await singleTokenGeneration(using: engine, context: context)

      expectNoDifference(parent.prefillMetrics.tokens, forked.prefillMetrics.tokens)
    }

    @Test
    func `Forking Beyond Sequence Capacity Falls Back To A Cold Cache`() async throws {
      let engine = try await qwen3LlamaEngine(
        contextParameters: LlamaContextParameters(maximumSequenceCount: 1)
      )
      let context = engine.context(llamaCachedContextParameters())
      let prefill = try await engine.prefill(context: context)

      let forked = try await singleTokenGeneration(using: engine, context: context.fork())
      let fresh = try await singleTokenGeneration(
        using: engine,
        context: engine.context(llamaCachedContextParameters())
      )

      expectNoDifference(prefill.metrics.tokens > 0, true)
      expectNoDifference(forked.prefillMetrics.tokens, fresh.prefillMetrics.tokens)
    }
  }

  // MARK: - Helpers

  private let llamaSystemPrompt = "You are a helpful assistant."
  private let llamaUserPrompt = "Say hello in one word."

  private func llamaCachedContextParameters() -> EdgeToolsTranscriptContextParameters {
    EdgeToolsTranscriptContextParameters(
      transcript: EdgeToolsTranscript(messages: [.system(llamaSystemPrompt)]),
      reasoningEffort: .none
    )
  }

  private func qwen3LlamaEngine(
    contextParameters: LlamaContextParameters = LlamaContextParameters()
  ) async throws -> Qwen3LlamaModelEngine {
    try Qwen3LlamaModelEngine(
      modelPath: (try await downloadGGUFModel(id: .qwen3)).path(),
      contextParameters: contextParameters
    )
  }

  private func singleTokenGeneration(
    using engine: Qwen3LlamaModelEngine,
    context: LlamaContext<Qwen3LlamaProfile>
  ) async throws -> EdgeToolsEngineGeneration {
    let task = try engine.generate(
      prompt: .user(llamaUserPrompt),
      tools: [],
      parameters: DefaultLlamaGenerateParameters(sampling: .greedy, maxTokens: 1),
      context: context,
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value
  }
#endif
