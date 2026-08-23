#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaContextState

  public struct LlamaContextState<Profile: LlamaModelProfile>: Sendable {
    let runtime: LlamaRuntime
    let sequence: LlamaSequenceLease
    let vocabularySizeValue: Int
    let configuredSampling: EdgeToolsFusedSamplingParameters?
    let preparedInputCache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput>

    init(
      runtime: LlamaRuntime,
      sequence: LlamaSequenceLease,
      vocabularySize: Int,
      defaultSampling: EdgeToolsFusedSamplingParameters?,
      preparedInputCache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput> =
        EdgeToolsLLMPreparedInputCache()
    ) {
      self.runtime = runtime
      self.sequence = sequence
      self.vocabularySizeValue = vocabularySize
      self.configuredSampling = defaultSampling
      self.preparedInputCache = preparedInputCache
    }

    var vocabularySize: Int {
      self.vocabularySizeValue
    }

    public func forkedContextState() -> sending Self {
      if let sequence = self.runtime.lease(copyingFrom: self.sequence.sequenceId) {
        return Self(
          runtime: self.runtime,
          sequence: sequence,
          vocabularySize: self.vocabularySizeValue,
          defaultSampling: self.configuredSampling,
          preparedInputCache: self.preparedInputCache.forked()
        )
      }
      let runtime = self.runtime.fresh()
      return Self(
        runtime: runtime,
        sequence: runtime.lease(copyingFrom: nil)!,
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.configuredSampling,
        preparedInputCache: self.preparedInputCache.forked()
      )
    }

    public func generationState() -> sending Self {
      self
    }
  }

  extension LlamaContextState: EdgeToolsForkableModelState {}

  // MARK: - LlamaInputProcessor

  struct LlamaInputProcessor<Profile: LlamaModelProfile>: Sendable {
    let tokenizer: LlamaTokenizer
    let multimodal: LlamaMultimodalInputProcessor<Profile>?

    var multimodalRuntime: LlamaMultimodalRuntime? {
      self.multimodal?.runtime
    }

    @concurrent
    func inputConcurrently(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      addGenerationPrompt: Bool,
      kind: EdgeToolsLLMInputKind,
      cache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput>
    ) async throws -> LlamaPreparedInput {
      try self.input(
        prompt: prompt,
        tools: tools,
        addGenerationPrompt: addGenerationPrompt,
        kind: kind,
        cache: cache
      )
    }

    func input(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      addGenerationPrompt: Bool,
      kind: EdgeToolsLLMInputKind,
      cache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput>
    ) throws -> LlamaPreparedInput {
      guard let multimodal else {
        guard prompt.images.isEmpty, prompt.videos.isEmpty, prompt.audio.isEmpty else {
          throw EdgeToolsError.unsupportedMedia(
            "This LlamaEngine was initialized without a multimodal runtime."
          )
        }
        if let cached = cache.input(for: prompt, tools: tools, kind: kind) {
          return cached
        }
        let input = LlamaPreparedInput(
          tokenIds: try Profile.tokenIds(
            prompt: prompt,
            tools: tools,
            tokenizer: self.tokenizer,
            addGenerationPrompt: addGenerationPrompt
          )
        )
        cache.store(input, for: prompt, tools: tools, kind: kind)
        return input
      }
      return try multimodal.input(
        prompt: prompt,
        tools: tools,
        addGenerationPrompt: addGenerationPrompt,
        kind: kind,
        cache: cache
      )
    }
  }

  // MARK: - LlamaGenerationTransaction

  struct LlamaGenerationTransaction<Profile: LlamaModelProfile> {
    typealias Decoder = DecoderState<EdgeToolsCPUFusedSampler>

    let contextState: LlamaContextState<Profile>
    var transcript: EdgeToolsTranscript
    let revision: Int
    var decoder: Decoder?
  }
#endif
