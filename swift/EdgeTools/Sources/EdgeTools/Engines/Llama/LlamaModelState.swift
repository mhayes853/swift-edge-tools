#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaModelState

  public struct LlamaModelState<Profile: LlamaModelProfile> {
    private struct Generation {
      var pendingTokenId: EdgeToolsToken.ID?
      let sampler: EdgeToolsCPUFusedSampler
      var confidence = ConfidenceState()
    }

    private let family: LlamaSequenceFamily
    private let lease: LlamaSequenceFamily.Lease
    private let vocabularySizeValue: Int
    private let configuredSampling: EdgeToolsFusedSamplingParameters?
    private var generation: Generation?

    init(
      family: LlamaSequenceFamily,
      lease: LlamaSequenceFamily.Lease,
      vocabularySize: Int,
      defaultSampling: EdgeToolsFusedSamplingParameters?
    ) {
      self.family = family
      self.lease = lease
      self.vocabularySizeValue = vocabularySize
      self.configuredSampling = defaultSampling
    }

    public var vocabularySize: Int {
      self.vocabularySizeValue
    }

    public func forkedContextState(copyingCache: Bool) -> sending Self {
      let state: Self
      if let lease = self.family.lease(copyingFrom: self.lease.sequenceId) {
        state = Self(
          family: self.family,
          lease: lease,
          vocabularySize: self.vocabularySizeValue,
          defaultSampling: self.configuredSampling
        )
      } else {
        let family = LlamaSequenceFamily(
          model: self.family.model,
          parameters: self.family.parameters,
          multimodalProjector: self.family.multimodalProjector
        )
        state = Self(
          family: family,
          lease: family.lease(copyingFrom: nil)!,
          vocabularySize: self.vocabularySizeValue,
          defaultSampling: self.configuredSampling
        )
      }
      nonisolated(unsafe) let transferredState = state
      return transferredState
    }

    public func generationState() -> sending Self {
      nonisolated(unsafe) let state = Self(
        family: self.family,
        lease: self.lease,
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.configuredSampling
      )
      return state
    }

    public func warmUp() throws {
      try self.family.warmUp()
    }

    public mutating func prepare(
      prompt: inout EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser
    ) throws -> EdgeToolsGenerationLoop.Preparation {
      Profile.prepare(prompt: &prompt, tools: tools, parser: &parser)
      let input = try self.preparedInput(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        addGenerationPrompt: true
      )
      self.family.resetProbe(sequenceId: self.lease.sequenceId)
      let metrics = try self.prefill(input: input, wantsLogits: true)
      let defaultSampling =
        Profile.defaultSampling(prompt: prompt, parameters: parameters)
        ?? self.configuredSampling
        ?? EdgeToolsFusedSamplingParameters()
      self.generation = Generation(
        sampler: EdgeToolsCPUFusedSampler(
          parameters: parameters.sampling.applying(to: defaultSampling)
        )
      )
      return EdgeToolsGenerationLoop.Preparation(metrics: metrics)
    }

    public mutating func prefill(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) throws -> EdgeToolsEnginePrefill {
      let input = try self.preparedInput(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        addGenerationPrompt: false
      )
      return EdgeToolsEnginePrefill(metrics: try self.prefill(input: input, wantsLogits: false))
    }

    public func tokenIds(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) throws -> [EdgeToolsToken.ID] {
      let input = try self.preparedInput(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        addGenerationPrompt: true
      )
      return input.units.tokenIds
    }

    public mutating func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters
    ) throws -> EdgeToolsToken.ID {
      guard let generation = self.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      let sample = try self.family.withCurrentLogits(
        sequenceId: self.lease.sequenceId,
        appending: generation.pendingTokenId,
        vocabularySize: self.vocabularySizeValue
      ) {
        generation.sampler.sample(logits: &$0, bitmask: bitmask)
      }
      self.generation?.confidence.add(confidence: sample.confidence)
      self.generation?.pendingTokenId = sample.tokenId
      return sample.tokenId
    }

    public mutating func commitGeneration(stopTokenIds: Set<EdgeToolsToken.ID>) {
      guard let generation = self.generation else { return }
      if let pendingTokenId = generation.pendingTokenId,
        !stopTokenIds.contains(pendingTokenId)
      {
        try? self.family.append(tokenId: pendingTokenId, sequenceId: self.lease.sequenceId)
      }
      self.generation = nil
    }

    public mutating func resetGeneration() {
      self.generation = nil
    }

    public func finish() -> EdgeToolsMetadata {
      guard let generation = self.generation else { return EdgeToolsMetadata() }
      var metadata = EdgeToolsMetadata()
      metadata.generationConfidence = generation.confidence.mean
      metadata.perTokenConfidences = generation.confidence.perTokenConfidences
      metadata.probeConfidence = self.family.probeConfidence(sequenceId: self.lease.sequenceId)
      return metadata
    }

    private func prefill(
      input: borrowing LlamaPreparedInput,
      wantsLogits: Bool
    ) throws -> EdgeToolsPrefillMetrics {
      let clock = ContinuousClock()
      let start = clock.now
      let decodedCount = try self.family.prefill(
        sequenceId: self.lease.sequenceId,
        input: input,
        wantsLogits: wantsLogits
      )
      return EdgeToolsPrefillMetrics(tokens: decodedCount, duration: start.duration(to: clock.now))
    }

    private func preparedInput(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      addGenerationPrompt: Bool
    ) throws -> LlamaPreparedInput {
      guard let projector = self.family.multimodalProjector else {
        guard prompt.images.isEmpty, prompt.videos.isEmpty, prompt.audio.isEmpty else {
          throw EdgeToolsError.unsupportedMedia(
            "This LlamaEngine was initialized without a multimodal projector."
          )
        }
        return LlamaPreparedInput(
          tokenIds: try Profile.tokenIds(
            prompt: prompt,
            tools: tools,
            tokenizer: tokenizer,
            addGenerationPrompt: addGenerationPrompt
          )
        )
      }
      #if FoundationEssentials
        guard prompt.videos.isEmpty, prompt.audio.isEmpty else {
          throw EdgeToolsError.unsupportedMedia(
            "This LlamaEngine integration supports image input only."
          )
        }
        guard
          let profile = Profile.self as? any EdgeToolsMultimodalModelProfile.Type,
          let tokenizer = tokenizer as? any EdgeToolsChatTokenizer
        else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        var images = [EdgeToolsTranscript.Asset]()
        let messages = try prompt.chatTemplateMessages { userMessage in
          let content = profile.multimodalContent(for: userMessage).reduce(into: "") {
            content, part in
            switch part {
            case .text(let text): content.append(text)
            case .image(let image):
              content.append(projector.mediaMarker)
              images.append(image)
            case .video, .audio:
              break
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
        return try projector.prepare(text: text, images: images)
      #else
        throw EdgeToolsError.unsupportedTokenizer
      #endif
    }
  }

  extension LlamaModelState: EdgeToolsForkableModelState {}

  // MARK: - LlamaGenerationState

  public struct LlamaGenerationState<Profile: LlamaModelProfile> {
    public var model: LlamaModelState<Profile>
    public var transcript: EdgeToolsTranscript
    public let revision: Int

    public init(
      model: LlamaModelState<Profile>,
      transcript: EdgeToolsTranscript,
      revision: Int
    ) {
      self.model = model
      self.transcript = transcript
      self.revision = revision
    }
  }
#endif
