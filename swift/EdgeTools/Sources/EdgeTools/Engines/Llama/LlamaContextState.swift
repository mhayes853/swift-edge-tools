#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if !$Embedded
    import Observation
  #endif

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaContext

  public final class LlamaContext: EdgeToolsEngineContext {
    typealias Snapshot = TranscriptContextStorage<LlamaContextState>.Snapshot

    let storage: TranscriptContextStorage<LlamaContextState>

    public var tools: [any EdgeTool] {
      self.storage.tools
    }

    public var transcript: EdgeToolsTranscript {
      get { self.storage.transcript }
      set { self.storage.transcript = newValue }
    }

    public var reasoningEffort: EdgeToolsReasoningEffort {
      get { self.storage.reasoningEffort }
      set { self.storage.reasoningEffort = newValue }
    }

    public var isResponding: Bool {
      self.storage.isResponding
    }

    init(
      transcript: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [any EdgeTool],
      model: sending LlamaContextState,
      engineIdentity: EdgeToolsEngineIdentity
    ) {
      self.storage = TranscriptContextStorage(
        transcript: transcript,
        reasoningEffort: reasoningEffort,
        tools: tools,
        model: model,
        engineIdentity: engineIdentity
      )
    }

    private init(storage: TranscriptContextStorage<LlamaContextState>) {
      self.storage = storage
    }

    public func fork() -> LlamaContext {
      LlamaContext(storage: self.storage.fork(model: { $0.forked() }))
    }
  }

  #if !$Embedded
    extension LlamaContext: Observable {}
  #endif

  // MARK: - LlamaContextState

  struct LlamaContextState: Sendable {
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

    func forked() -> sending Self {
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
  }

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
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      addGenerationPrompt: Bool,
      kind: EdgeToolsLLMInputKind,
      cache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput>
    ) async throws -> LlamaPreparedInput {
      try self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        addGenerationPrompt: addGenerationPrompt,
        kind: kind,
        cache: cache
      )
    }

    func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
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
        if let cached = cache.input(
          for: prompt,
          reasoningEffort: reasoningEffort,
          tools: tools,
          kind: kind
        ) {
          return cached
        }
        let input = LlamaPreparedInput(
          tokenIds: try Profile.tokenIds(
            prompt: prompt,
            reasoningEffort: reasoningEffort,
            tools: tools,
            tokenizer: self.tokenizer,
            addGenerationPrompt: addGenerationPrompt
          )
        )
        cache.store(
          input,
          for: prompt,
          reasoningEffort: reasoningEffort,
          tools: tools,
          kind: kind
        )
        return input
      }
      return try multimodal.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        addGenerationPrompt: addGenerationPrompt,
        kind: kind,
        cache: cache
      )
    }
  }

  // MARK: - LlamaGenerationTransaction

  struct LlamaGenerationTransaction {
    typealias Decoder = DecoderState<EdgeToolsCPUFusedSampler>

    let contextState: LlamaContextState
    var transcript: EdgeToolsTranscript
    let reasoningEffort: EdgeToolsReasoningEffort
    let revision: Int
    var decoder: Decoder?
  }
#endif
