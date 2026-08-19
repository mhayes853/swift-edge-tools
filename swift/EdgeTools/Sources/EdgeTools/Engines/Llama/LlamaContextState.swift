#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaContextState

  public struct LlamaContextState<Profile: LlamaModelProfile>: Sendable {
    let sequenceStore: LlamaKVSequenceStore
    let sequence: LlamaSequenceLease
    let vocabularySizeValue: Int
    let configuredSampling: EdgeToolsFusedSamplingParameters?
    let preparedInputCache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput>

    init(
      sequenceStore: LlamaKVSequenceStore,
      sequence: LlamaSequenceLease,
      vocabularySize: Int,
      defaultSampling: EdgeToolsFusedSamplingParameters?,
      preparedInputCache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput> =
        EdgeToolsLLMPreparedInputCache()
    ) {
      self.sequenceStore = sequenceStore
      self.sequence = sequence
      self.vocabularySizeValue = vocabularySize
      self.configuredSampling = defaultSampling
      self.preparedInputCache = preparedInputCache
    }

    public var vocabularySize: Int {
      self.vocabularySizeValue
    }

    public func forkedContextState(copyingCache: Bool) -> sending Self {
      if let sequence = self.sequenceStore.lease(copyingFrom: self.sequence.sequenceId) {
        return Self(
          sequenceStore: self.sequenceStore,
          sequence: sequence,
          vocabularySize: self.vocabularySizeValue,
          defaultSampling: self.configuredSampling,
          preparedInputCache: self.preparedInputCache.forked()
        )
      }
      let sequenceStore = LlamaKVSequenceStore(
        model: self.sequenceStore.model,
        parameters: self.sequenceStore.parameters
      )
      return Self(
        sequenceStore: sequenceStore,
        sequence: sequenceStore.lease(copyingFrom: nil)!,
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
    let multimodalRuntime: LlamaMultimodalRuntime?

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
      guard let multimodalRuntime else {
        guard prompt.images.isEmpty, prompt.videos.isEmpty, prompt.audio.isEmpty else {
          throw EdgeToolsError.unsupportedMedia(
            "This LlamaEngine was initialized without a multimodal runtime."
          )
        }
        return LlamaPreparedInput(
          tokenIds: try Profile.tokenIds(
            prompt: prompt,
            tools: tools,
            tokenizer: self.tokenizer,
            addGenerationPrompt: addGenerationPrompt
          )
        )
      }
      guard prompt.videos.isEmpty else {
        throw EdgeToolsError.unsupportedMedia("Video input is not supported by LlamaEngine.")
      }
      guard
        let profile = Profile.self as? any EdgeToolsMultimodalModelProfile.Type
      else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let tokenizer = self.tokenizer
      var media = [LlamaMultimodalAsset]()
      let messages = try prompt.chatTemplateMessages { userMessage in
        let content = try profile.multimodalContent(for: userMessage)
          .reduce(into: "") { content, part in
            switch part {
            case .text(let text):
              content.append(text)
            case .image(let image):
              content.append(multimodalRuntime.mediaMarker)
              media.append(LlamaMultimodalAsset(kind: .image, asset: image))
            case .audio(let audio):
              content.append(multimodalRuntime.mediaMarker)
              media.append(LlamaMultimodalAsset(kind: .audio, asset: audio))
            case .video:
              throw EdgeToolsError.unsupportedMedia(
                "Video input is not supported by LlamaEngine."
              )
            }
          }
        return ["role": "user", "content": .string(content)]
      }
      let text = try tokenizer.renderChatTemplate(
        messages: messages,
        tools: tools.chatTemplateToolValues,
        addGenerationPrompt: addGenerationPrompt,
        additionalContext: Profile.templateContext(prompt: prompt)
      )
      let cachedInput = cache.input(
        for: prompt,
        tools: tools,
        kind: kind,
        allowingTextOnlyContinuation: true
      )
      if let cached = cachedInput,
        let input = try cached.replacingText(
          text,
          mediaMarker: multimodalRuntime.mediaMarker,
          tokenizer: tokenizer
        )
      {
        cache.store(input, for: prompt, tools: tools, kind: kind)
        return input
      }
      let input = try multimodalRuntime.prepare(text: text, media: media)
      cache.store(input, for: prompt, tools: tools, kind: kind)
      return input
    }
  }

  // MARK: - LlamaGenerationTransaction

  struct LlamaGenerationTransaction<Profile: LlamaModelProfile> {
    struct Decoder {
      var pendingTokenId: EdgeToolsToken.ID?
      let sampler: EdgeToolsCPUFusedSampler
      var confidence = ConfidenceState()
    }

    let contextState: LlamaContextState<Profile>
    var transcript: EdgeToolsTranscript
    let revision: Int
    var decoder: Decoder?
  }
#endif
