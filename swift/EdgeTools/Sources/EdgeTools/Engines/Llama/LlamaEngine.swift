#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if FoundationEssentials
    import _EdgeToolsFoundation
  #endif

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaEngine

  public final class LlamaEngine<Profile: LlamaModelProfile>:
    EdgeToolsEngine, EdgeToolsPrefillableEngine, EdgeToolsTokenizingEngine {
    public typealias Context = LlamaContext<Profile>
    public typealias ContextParameters = EdgeToolsTranscriptContextParameters
    public typealias Prompt = EdgeToolsTranscript.Prompt
    public typealias GenerateParameters = Profile.GenerateParameters
    typealias ModelGenerationState = LlamaGenerationTransaction<Profile>

    private let model: LlamaModelBox
    private let multimodalRuntime: LlamaMultimodalRuntime?
    private let contextParameters: LlamaContextParameters
    private let defaultSampling: EdgeToolsFusedSamplingParameters?
    private let vocabularySizeValue: Int
    private let identity = EdgeToolsEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
    private let inputProcessor: LlamaInputProcessor<Profile>
    public let tokenizer: any EdgeToolsTokenizer
    public let grammarEngine: Profile.GrammarEngine

    public convenience init(
      modelPath: String,
      modelParameters: LlamaModelParameters = LlamaModelParameters(),
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      try self.init(
        model: LlamaModel(path: modelPath, parameters: modelParameters),
        projectorPath: nil,
        contextParameters: contextParameters,
        multimodalParameters: LlamaMultimodalParameters(),
        defaultSampling: defaultSampling
      )
    }

    public convenience init(
      model: consuming LlamaModel,
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      try self.init(
        model: consume model,
        projectorPath: nil,
        contextParameters: contextParameters,
        multimodalParameters: LlamaMultimodalParameters(),
        defaultSampling: defaultSampling
      )
    }

    public convenience init(
      modelPath: String,
      multimodalProjectorPath: String,
      modelParameters: LlamaModelParameters = LlamaModelParameters(),
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      multimodalParameters: LlamaMultimodalParameters = LlamaMultimodalParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws where Profile: EdgeToolsMultimodalModelProfile {
      try self.init(
        model: LlamaModel(path: modelPath, parameters: modelParameters),
        projectorPath: multimodalProjectorPath,
        contextParameters: contextParameters,
        multimodalParameters: multimodalParameters,
        defaultSampling: defaultSampling
      )
    }

    public convenience init(
      model: consuming LlamaModel,
      multimodalProjectorPath: String,
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      multimodalParameters: LlamaMultimodalParameters = LlamaMultimodalParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws where Profile: EdgeToolsMultimodalModelProfile {
      try self.init(
        model: consume model,
        projectorPath: multimodalProjectorPath,
        contextParameters: contextParameters,
        multimodalParameters: multimodalParameters,
        defaultSampling: defaultSampling
      )
    }

    private init(
      model: consuming LlamaModel,
      projectorPath: String?,
      contextParameters: LlamaContextParameters,
      multimodalParameters: LlamaMultimodalParameters,
      defaultSampling: EdgeToolsFusedSamplingParameters?
    ) throws {
      let model = LlamaModelBox(model: consume model)
      self.model = model
      let multimodalRuntime = try projectorPath.map {
        try LlamaMultimodalRuntime(path: $0, model: model, parameters: multimodalParameters)
      }
      self.multimodalRuntime = multimodalRuntime
      self.contextParameters = contextParameters
      self.defaultSampling = defaultSampling

      let tokenizer = LlamaTokenizer(model: model)
      self.tokenizer = tokenizer
      self.inputProcessor = LlamaInputProcessor(
        tokenizer: tokenizer,
        multimodalRuntime: multimodalRuntime
      )
      self.vocabularySizeValue = tokenizer.vocabularySize

      var stopTokenIds = tokenizer.endOfGenerationTokenIds()
      stopTokenIds.formUnion(
        Profile.extraStopTokens.compactMap { tokenizer.token(forText: $0)?.id }
      )
      self.grammarEngine = try Profile.grammarEngine(
        tokenizer: tokenizer,
        vocabularySize: self.vocabularySizeValue,
        stopTokenIds: stopTokenIds
      )
      self.generationLoop = EdgeToolsGenerationLoop(
        tokenizer: tokenizer,
        extraStopTokenIds: stopTokenIds
      )
    }

    public func context() -> LlamaContext<Profile> {
      self.context(EdgeToolsTranscriptContextParameters())
    }

    public func context(
      _ parameters: EdgeToolsTranscriptContextParameters
    ) -> LlamaContext<Profile> {
      EdgeToolsTranscriptContext(
        parameters: parameters,
        model: self.contextState(),
        engineIdentity: self.identity
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.Prompt,
      tools: [EdgeToolDefinition],
      context: LlamaContext<Profile>
    ) async throws -> [EdgeToolsToken] {
      try self.validate(context)
      let transcript = context.transcript(appending: prompt)
      let input = try self.inputProcessor.input(
        prompt: transcript,
        tools: tools,
        addGenerationPrompt: true
      )
      return self.tokenizer.tokens(forIds: input.units.tokenIds).compactMap { $0 }
    }

    public func generate(
      prompt: EdgeToolsTranscript.Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: LlamaContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: { self.generationState(from: try context.begin(appending: prompt)) }
      )
    }

    public func prefill(
      promptPrefix: EdgeToolsTranscript.Prompt,
      tools: [EdgeToolDefinition],
      context: LlamaContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      let snapshot = try context.begin(appending: promptPrefix)
      return try self.prefill(snapshot: snapshot, tools: tools, context: context)
    }

    public func warmUp(tools: [EdgeToolDefinition] = [], context: LlamaContext<Profile>) throws {
      try self.validate(context)
      let snapshot = try context.begin()
      defer { context.finish(generation: nil, revision: snapshot.revision, model: snapshot.model) }
      try snapshot.model.sequenceStore.warmUp()
      _ = try self.inputProcessor.input(
        prompt: snapshot.transcript,
        tools: tools,
        addGenerationPrompt: true
      )
    }

    public func prefill(
      tools: [EdgeToolDefinition] = [],
      context: LlamaContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try self.prefill(
        snapshot: context.begin(),
        tools: tools,
        context: context
      )
    }

    private func generationTask(
      tools: [EdgeToolDefinition],
      parameters: sending Profile.GenerateParameters,
      context: LlamaContext<Profile>,
      channel: sending EdgeToolsGenerationChannel,
      makeState: @escaping @Sendable () throws -> ModelGenerationState
    ) -> AnyGenerationTask {
      AnyGenerationTask { stopper in
        var state = try makeState()
        let result: Result<EdgeToolsEngineGeneration, any Error>
        do {
          result = .success(
            try await self.generationLoop.run(
              state: &state,
              stopper: stopper,
              channel: channel,
              grammarEngine: self.grammarEngine,
              maximumTokenCount: parameters.maxTokens,
              grammar: {
                try Profile.grammar(
                  prompt: $0.transcript,
                  tools: tools,
                  parameters: parameters,
                  grammarEngine: self.grammarEngine
                )
              },
              prepare: {
                try self.prepare(parser: &$0, tools: tools, parameters: parameters, state: &$1)
              },
              decode: { try self.decode(bitmask: $0, state: &$1) }
            )
          )
        } catch {
          result = .failure(error)
        }
        return try self.finalize(state: state, result: result, context: context).get()
      }
    }

    private func prepare(
      parser: inout Profile.GenerationParser,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      state: inout ModelGenerationState
    ) throws -> EdgeToolsGenerationLoop.Preparation {
      Profile.prepare(prompt: &state.transcript, tools: tools, parser: &parser)
      let input = try self.inputProcessor.input(
        prompt: state.transcript,
        tools: tools,
        addGenerationPrompt: true
      )
      let contextState = state.contextState
      contextState.sequenceStore.resetProbe(sequenceId: contextState.sequence.sequenceId)
      let metrics = try self.synchronize(
        input: input,
        contextState: contextState,
        output: .lastTokenLogits
      )
      let defaultSampling =
        Profile.defaultSampling(prompt: state.transcript, parameters: parameters)
        ?? contextState.configuredSampling
        ?? EdgeToolsFusedSamplingParameters()
      state.decoder = ModelGenerationState.Decoder(
        sampler: EdgeToolsCPUFusedSampler(
          parameters: parameters.sampling.applying(to: defaultSampling)
        )
      )
      return EdgeToolsGenerationLoop.Preparation(metrics: metrics)
    }

    private func decode(
      bitmask: GrammarBitmask,
      state: inout ModelGenerationState
    ) throws -> EdgeToolsToken.ID {
      guard var decoder = state.decoder else {
        throw EdgeToolsError.modelNotPrepared
      }
      let contextState = state.contextState
      let sample = try contextState.sequenceStore.withLogits(
        sequenceId: contextState.sequence.sequenceId,
        appending: decoder.pendingTokenId,
        vocabularySize: contextState.vocabularySizeValue
      ) {
        decoder.sampler.sample(logits: &$0, bitmask: bitmask)
      }
      decoder.confidence.add(confidence: sample.confidence)
      decoder.pendingTokenId = sample.tokenId
      state.decoder = decoder
      return sample.tokenId
    }

    private func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: LlamaContext<Profile>
    ) -> Result<EdgeToolsEngineGeneration, any Error> {
      var state = state
      let metadata = self.metadata(for: state)

      let finalResult: Result<EdgeToolsEngineGeneration, any Error>
      switch result {
      case .success(var value):
        self.commitGeneration(state: &state)
        value.metadata.merge(metadata) { _, finalValue in finalValue }
        finalResult = .success(value)
      case .failure(let error):
        state.decoder = nil
        finalResult = .failure(error)
      }
      context.finish(
        generation: try? finalResult.get(),
        revision: state.revision,
        model: state.contextState
      )
      return finalResult
    }

    private func prefill(
      snapshot: LlamaContext<Profile>.Snapshot,
      tools: [EdgeToolDefinition],
      context: LlamaContext<Profile>
    ) throws -> EdgeToolsEnginePrefill {
      defer {
        context.finish(generation: nil, revision: snapshot.revision, model: snapshot.model)
      }
      let input = try self.inputProcessor.input(
        prompt: snapshot.transcript,
        tools: tools,
        addGenerationPrompt: false
      )
      return EdgeToolsEnginePrefill(
        metrics: try self.synchronize(
          input: input,
          contextState: snapshot.model,
          output: .none
        )
      )
    }

    private func synchronize(
      input: borrowing LlamaPreparedInput,
      contextState: LlamaContextState<Profile>,
      output: LlamaEvaluationOutput
    ) throws -> EdgeToolsPrefillMetrics {
      let clock = ContinuousClock()
      let start = clock.now
      let evaluatedCount = try contextState.sequenceStore.synchronize(
        sequenceId: contextState.sequence.sequenceId,
        input: input,
        multimodalRuntime: self.multimodalRuntime,
        output: output
      )
      return EdgeToolsPrefillMetrics(
        tokens: evaluatedCount,
        duration: start.duration(to: clock.now)
      )
    }

    private func metadata(for state: ModelGenerationState) -> EdgeToolsMetadata {
      guard let decoder = state.decoder else { return EdgeToolsMetadata() }
      var metadata = EdgeToolsMetadata()
      metadata.generationConfidence = decoder.confidence.mean
      metadata.perTokenConfidences = decoder.confidence.perTokenConfidences
      metadata.probeConfidence = state.contextState.sequenceStore.probeConfidence(
        sequenceId: state.contextState.sequence.sequenceId
      )
      return metadata
    }

    private func commitGeneration(state: inout ModelGenerationState) {
      guard let decoder = state.decoder else { return }
      if let pendingTokenId = decoder.pendingTokenId,
        !self.generationLoop.stopTokenIds.contains(pendingTokenId)
      {
        try? state.contextState.sequenceStore.commit(
          tokenId: pendingTokenId,
          sequenceId: state.contextState.sequence.sequenceId
        )
      }
      state.decoder = nil
    }

    private func contextState() -> sending LlamaContextState<Profile> {
      let sequenceStore = LlamaKVSequenceStore(
        model: self.model,
        parameters: self.contextParameters
      )
      return LlamaContextState(
        sequenceStore: sequenceStore,
        sequence: sequenceStore.lease(copyingFrom: nil)!,
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.defaultSampling
      )
    }

    private func validate(_ context: LlamaContext<Profile>) throws {
      guard context.engineIdentity === self.identity else {
        throw EdgeToolsError.incompatibleContext
      }
    }

    private func generationState(
      from snapshot: LlamaContext<Profile>.Snapshot
    ) -> ModelGenerationState {
      ModelGenerationState(
        contextState: snapshot.model,
        transcript: snapshot.transcript,
        revision: snapshot.revision
      )
    }
  }

  // MARK: - Foundation Essentials

  #if FoundationEssentials
    extension LlamaEngine {
      public convenience init(
        modelURL: URL,
        modelParameters: LlamaModelParameters = LlamaModelParameters(),
        contextParameters: LlamaContextParameters = LlamaContextParameters(),
        defaultSampling: EdgeToolsFusedSamplingParameters? = nil
      ) throws {
        try self.init(
          modelPath: modelURL.path(),
          modelParameters: modelParameters,
          contextParameters: contextParameters,
          defaultSampling: defaultSampling
        )
      }

      public convenience init(
        modelURL: URL,
        multimodalProjectorURL: URL,
        modelParameters: LlamaModelParameters = LlamaModelParameters(),
        contextParameters: LlamaContextParameters = LlamaContextParameters(),
        multimodalParameters: LlamaMultimodalParameters = LlamaMultimodalParameters(),
        defaultSampling: EdgeToolsFusedSamplingParameters? = nil
      ) throws where Profile: EdgeToolsMultimodalModelProfile {
        try self.init(
          modelPath: modelURL.path(),
          multimodalProjectorPath: multimodalProjectorURL.path(),
          modelParameters: modelParameters,
          contextParameters: contextParameters,
          multimodalParameters: multimodalParameters,
          defaultSampling: defaultSampling
        )
      }

      public convenience init(
        model: consuming LlamaModel,
        multimodalProjectorURL: URL,
        contextParameters: LlamaContextParameters = LlamaContextParameters(),
        multimodalParameters: LlamaMultimodalParameters = LlamaMultimodalParameters(),
        defaultSampling: EdgeToolsFusedSamplingParameters? = nil
      ) throws where Profile: EdgeToolsMultimodalModelProfile {
        try self.init(
          model: consume model,
          multimodalProjectorPath: multimodalProjectorURL.path(),
          contextParameters: contextParameters,
          multimodalParameters: multimodalParameters,
          defaultSampling: defaultSampling
        )
      }
    }
  #endif

  // MARK: - EdgeToolsSession + Llama

  extension EdgeToolsSession {
    public func context<Profile>(
      transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> LlamaContext<Profile> where Engine == LlamaEngine<Profile> {
      self.context(
        EdgeToolsTranscriptContextParameters(
          transcript: transcript,
          reasoningEffort: reasoningEffort
        )
      )
    }

    public func context<Profile>(
      systemPrompt: String,
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> LlamaContext<Profile> where Engine == LlamaEngine<Profile> {
      let transcript = EdgeToolsTranscript(messages: [.system(systemPrompt)])
      return self.context(transcript: transcript, reasoningEffort: reasoningEffort)
    }
  }
#endif
