#if HuggingFaceTokenizers && Llama && XGrammar && canImport(CLlama) && !os(WASI)
  import CustomDump
  @testable import EdgeTools
  import EdgeToolsLlama
  import EdgeToolsXGrammar
  import Testing

  @Suite(.serialized)
  struct `LlamaEngine tests` {
    @Test
    func `A Prefill Without Logits Invalidates Another Sequences Logits`() async throws {
      let model = try LlamaModel(path: (try await downloadGGUFModel(id: .qwen3)).path())
      let modelBox = LlamaModelBox(model: consume model)
      let store = LlamaKVSequenceStore(
        model: modelBox,
        parameters: LlamaContextParameters(cacheForking: .copyOnWrite(maxContexts: 2))
      )
      let first = try #require(store.lease(copyingFrom: nil))
      let second = try #require(store.lease(copyingFrom: nil))
      let input = LlamaPreparedInput(tokenIds: [1])

      _ = try store.synchronizeForLogits(
        sequenceId: first.sequenceId,
        input: input,
        multimodalRuntime: nil
      )
      _ = try store.synchronize(
        sequenceId: second.sequenceId,
        input: input,
        multimodalRuntime: nil
      )

      #expect(throws: Never.self) {
        try store.withLogits(
          sequenceId: first.sequenceId,
          appending: nil,
          vocabularySize: 1
        ) { _ in }
      }
    }

    @Test
    func `A Failed Final Cache Commit Fails The Generation`() async throws {
      let modelPath = (try await downloadGGUFModel(id: .qwen3)).path()
      let filledContext = try llamaPromptFillingContext(modelPath: modelPath)
      let engine = try qwen3LlamaEngine(
        modelPath: modelPath,
        contextParameters: LlamaContextParameters(
          contextLength: UInt32(filledContext.tokenCount),
          cacheForking: .isolated
        )
      )
      let emittedTokenCount = Lock(0)
      let task = try engine.generate(
        prompt: .user(filledContext.prompt),
        parameters: LlamaGenerateParameters(sampling: .greedy, maxTokens: 1),
        context: engine.context(),
        channel: EdgeToolsGenerationChannel(
          onToken: { _ in emittedTokenCount.withLock { $0 += 1 } }
        )
      )

      await #expect(throws: LlamaRuntimeError.self) {
        try await task.value
      }
      expectNoDifference(emittedTokenCount.withLock { $0 }, 1)
    }

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
        (cached.metrics.prefillTokens ?? 0) + (prefill.metrics.prefillTokens ?? 0),
        fresh.metrics.prefillTokens ?? 0
      )
    }

    @Test
    func `Forking Leaves The Parent Cache Intact`() async throws {
      let engine = try await qwen3LlamaEngine()
      let context = engine.context(llamaCachedContextParameters())
      _ = try await engine.prefill(context: context)

      let forked = try await singleTokenGeneration(using: engine, context: context.fork())
      let parent = try await singleTokenGeneration(using: engine, context: context)

      expectNoDifference(parent.metrics.prefillTokens, forked.metrics.prefillTokens)
    }

    @Test
    func `Forking Beyond Sequence Capacity Falls Back To A Cold Cache`() async throws {
      let engine = try await qwen3LlamaEngine(
        contextParameters: LlamaContextParameters(cacheForking: .isolated)
      )
      let context = engine.context(llamaCachedContextParameters())
      let prefill = try await engine.prefill(context: context)

      let forked = try await singleTokenGeneration(using: engine, context: context.fork())
      let fresh = try await singleTokenGeneration(
        using: engine,
        context: engine.context(llamaCachedContextParameters())
      )

      expectNoDifference((prefill.metrics.prefillTokens ?? 0) > 0, true)
      expectNoDifference(forked.metrics.prefillTokens, fresh.metrics.prefillTokens)
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
    try qwen3LlamaEngine(
      modelPath: (try await downloadGGUFModel(id: .qwen3)).path(),
      contextParameters: contextParameters
    )
  }

  private func qwen3LlamaEngine(
    modelPath: String,
    contextParameters: LlamaContextParameters = LlamaContextParameters()
  ) throws -> Qwen3LlamaModelEngine {
    try Qwen3LlamaModelEngine(
      modelPath: modelPath,
      contextParameters: contextParameters
    )
  }

  private func llamaPromptFillingContext(
    modelPath: String
  ) throws -> (prompt: String, tokenCount: Int) {
    let tokenizer = LlamaTokenizer(model: try LlamaModel(path: modelPath))
    for wordCount in 1...512 {
      let prompt = String(repeating: " word", count: wordCount)
      let tokenCount =
        try Qwen3LlamaProfile.tokenIds(
          prompt: EdgeToolsTranscript(messages: [.user(prompt)]),
          tools: [],
          tokenizer: tokenizer,
          addGenerationPrompt: true
        )
        .count
      if tokenCount.isMultiple(of: 256) {
        return (prompt, tokenCount)
      }
    }
    throw LlamaRuntimeError(
      code: .contextCreationFailed,
      message: "A prompt aligned to the llama context size could not be found."
    )
  }

  private func singleTokenGeneration(
    using engine: Qwen3LlamaModelEngine,
    context: LlamaContext<Qwen3LlamaProfile>
  ) async throws -> EdgeToolsEngineGeneration {
    let task = try engine.generate(
      prompt: .user(llamaUserPrompt),
      parameters: LlamaGenerateParameters(sampling: .greedy, maxTokens: 1),
      context: context,
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value
  }
#endif
