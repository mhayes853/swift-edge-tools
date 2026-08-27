#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsLlama
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
    public typealias Context = LlamaContext
    public typealias ContextParameters = EdgeToolsTranscript
    public typealias Prompt = EdgeToolsTranscript.Prompt
    public typealias GenerateParameters = Profile.GenerateParameters
    typealias ModelGenerationState = LlamaGenerationTransaction

    private let model: LlamaModelBox
    private let contextParameters: LlamaContextParameters
    private let defaultSampling: EdgeToolsFusedSamplingParameters?
    private let vocabularySizeValue: Int
    private let identity = EdgeToolsEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
    private let inputProcessor: LlamaInputProcessor<Profile>
    private let tokenizer: any EdgeToolsTokenizer
    private let grammarEngine: Profile.GrammarEngine

    public convenience init(
      modelPath: String,
      modelParameters: LlamaModelParameters = LlamaModelParameters(),
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      LlamaBackend.initialize()
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
      LlamaBackend.initialize()
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
      LlamaBackend.initialize()
      let model = LlamaModelBox(model: consume model)
      self.model = model
      let multimodalRuntime = try projectorPath.map {
        try LlamaMultimodalRuntime(path: $0, model: model, parameters: multimodalParameters)
      }
      self.contextParameters = contextParameters
      self.defaultSampling = defaultSampling

      let tokenizer = LlamaTokenizer(model: model)
      self.tokenizer = tokenizer
      self.inputProcessor = LlamaInputProcessor(
        tokenizer: tokenizer,
        multimodal: multimodalRuntime.map {
          LlamaMultimodalInputProcessor(tokenizer: tokenizer, runtime: $0)
        }
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

    public func context(tools: [any EdgeTool]) -> LlamaContext {
      self.context(
        transcript: EdgeToolsTranscript(),
        reasoningEffort: .default,
        tools: tools
      )
    }

    public func context(
      _ transcript: EdgeToolsTranscript,
      tools: [any EdgeTool]
    ) -> LlamaContext {
      self.context(transcript: transcript, reasoningEffort: .default, tools: tools)
    }

    public func context(
      transcript: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [any EdgeTool] = []
    ) -> LlamaContext {
      LlamaContext(
        transcript: transcript,
        reasoningEffort: reasoningEffort,
        tools: tools,
        model: self.contextState(),
        engineIdentity: self.identity
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.Prompt,
      context: LlamaContext
    ) async throws -> [EdgeToolsToken] {
      try self.validate(context)
      let prompt = context.storage.prompt(appending: prompt)
      let input = try await self.inputProcessor.inputConcurrently(
        prompt: prompt.transcript,
        reasoningEffort: prompt.reasoningEffort,
        tools: context.tools.map { $0.definition },
        addGenerationPrompt: true,
        kind: .generation,
        cache: EdgeToolsLLMPreparedInputCache()
      )
      return self.tokenizer.tokens(forIds: input.units.tokenIds).compactMap { $0 }
    }

    public func generate(
      prompt: EdgeToolsTranscript.Prompt,
      parameters: sending Profile.GenerateParameters,
      context: LlamaContext,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        tools: context.tools.map { $0.definition },
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: { self.generationState(from: try context.storage.begin(appending: prompt)) }
      )
    }

    public func prefill(
      promptPrefix: EdgeToolsTranscript.Prompt,
      context: LlamaContext
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      let snapshot = try context.storage.begin(appending: promptPrefix)
      return try await self.prefill(
        snapshot: snapshot,
        tools: context.tools.map { $0.definition },
        context: context
      )
    }

    public func warmUp(context: LlamaContext) throws {
      try self.validate(context)
      let snapshot = try context.storage.begin()
      defer {
        context.storage.finish(generation: nil, revision: snapshot.revision, model: snapshot.model)
      }
      try snapshot.model.runtime.sequences.warmUp()
      _ = try self.inputProcessor.input(
        prompt: snapshot.transcript,
        reasoningEffort: snapshot.reasoningEffort,
        tools: context.tools.map { $0.definition },
        addGenerationPrompt: true,
        kind: .generation,
        cache: snapshot.model.preparedInputCache
      )
    }

    public func prefill(context: LlamaContext) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try await self.prefill(
        snapshot: context.storage.begin(),
        tools: context.tools.map { $0.definition },
        context: context
      )
    }

    private func generationTask(
      tools: [EdgeToolDefinition],
      parameters: sending Profile.GenerateParameters,
      context: LlamaContext,
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
                  reasoningEffort: $0.reasoningEffort,
                  tools: tools,
                  parameters: parameters,
                  grammarEngine: self.grammarEngine
                )
              },
              prepare: {
                try await self.prepare(parser: &$0, tools: tools, parameters: parameters, state: &$1)
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
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
      Profile.prepare(
        prompt: &state.transcript,
        reasoningEffort: state.reasoningEffort,
        tools: tools,
        parser: &parser
      )
      let input = try await self.inputProcessor.inputConcurrently(
        prompt: state.transcript,
        reasoningEffort: state.reasoningEffort,
        tools: tools,
        addGenerationPrompt: true,
        kind: .generation,
        cache: state.contextState.preparedInputCache
      )
      let contextState = state.contextState
      if parameters.confidence.contains(.probe) {
        contextState.runtime.sequences.resetProbe(sequenceId: contextState.sequence.sequenceId)
      }
      let metrics = try self.metrics {
        try contextState.runtime.sequences.synchronizeForLogits(
          sequenceId: contextState.sequence.sequenceId,
          input: input,
          multimodalRuntime: self.inputProcessor.multimodalRuntime
        )
      }
      let defaultSampling =
        Profile.defaultSampling(
          prompt: state.transcript,
          reasoningEffort: state.reasoningEffort,
          parameters: parameters
        )
        ?? contextState.configuredSampling
        ?? EdgeToolsFusedSamplingParameters()

      let sampler = EdgeToolsCPUFusedSampler(
        parameters: parameters.sampling.applying(to: defaultSampling)
      )
      state.decoder = ModelGenerationState.Decoder(
        sampler: sampler,
        confidenceOptions: parameters.confidence
      )
      return EdgeToolsGenerationLoop.Preparation(metrics: metrics)
    }

    private func decode(
      bitmask: GrammarBitmask?,
      state: inout ModelGenerationState
    ) throws -> EdgeToolsToken.ID {
      guard var decoder = state.decoder else { throw EdgeToolsError.modelNotPrepared }
      let contextState = state.contextState
      let sample = try contextState.runtime.sequences.withLogits(
        sequenceId: contextState.sequence.sequenceId,
        appending: decoder.pendingTokenId,
        vocabularySize: contextState.vocabularySizeValue
      ) {
        decoder.sampler.sample(logits: &$0, bitmask: bitmask)
      }
      decoder.add(confidence: sample.confidence)
      decoder.pendingTokenId = sample.tokenId
      state.decoder = decoder
      return sample.tokenId
    }

    private func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: LlamaContext
    ) -> Result<EdgeToolsEngineGeneration, any Error> {
      var state = state
      let metadata = self.metadata(for: state)

      let finalResult: Result<EdgeToolsEngineGeneration, any Error>
      switch result {
      case .success(var value):
        do {
          try self.commitGeneration(state: &state)
          value.metrics.merge(metadata) { _, finalValue in finalValue }
          finalResult = .success(value)
        } catch {
          state.decoder = nil
          finalResult = .failure(error)
        }
      case .failure(let error):
        state.decoder = nil
        finalResult = .failure(error)
      }
      context.storage.finish(
        generation: try? finalResult.get(),
        revision: state.revision,
        model: state.contextState
      )
      return finalResult
    }

    private func prefill(
      snapshot: LlamaContext.Snapshot,
      tools: [EdgeToolDefinition],
      context: LlamaContext
    ) async throws -> EdgeToolsEnginePrefill {
      defer {
        context.storage.finish(generation: nil, revision: snapshot.revision, model: snapshot.model)
      }
      let input = try await self.inputProcessor.inputConcurrently(
        prompt: snapshot.transcript,
        reasoningEffort: snapshot.reasoningEffort,
        tools: tools,
        addGenerationPrompt: false,
        kind: .prefill,
        cache: snapshot.model.preparedInputCache
      )
      return EdgeToolsEnginePrefill(
        metrics: try self.metrics {
          try snapshot.model.runtime.sequences.synchronize(
            sequenceId: snapshot.model.sequence.sequenceId,
            input: input,
            multimodalRuntime: self.inputProcessor.multimodalRuntime
          )
        }
      )
    }

    private func metrics(evaluating: () throws -> Int) rethrows -> EdgeToolsMetrics {
      let clock = ContinuousClock()
      let start = clock.now
      let evaluatedCount = try evaluating()
      var metrics = EdgeToolsMetrics()
      metrics.prefillTokens = evaluatedCount
      metrics.prefillDuration = start.duration(to: clock.now)
      return metrics
    }

    private func metadata(for state: ModelGenerationState) -> EdgeToolsMetrics {
      guard let decoder = state.decoder else { return EdgeToolsMetrics() }
      var metadata = decoder.metrics
      if decoder.confidenceOptions.contains(.probe) {
        metadata.probeConfidence = state.contextState.runtime.sequences.probeConfidence(
          sequenceId: state.contextState.sequence.sequenceId
        )
      }
      return metadata
    }

    private func commitGeneration(state: inout ModelGenerationState) throws {
      guard let decoder = state.decoder else { return }
      if let pendingTokenId = decoder.pendingTokenId,
        !self.generationLoop.stopTokenIds.contains(pendingTokenId)
      {
        try state.contextState.runtime.sequences.commit(
          tokenId: pendingTokenId,
          sequenceId: state.contextState.sequence.sequenceId
        )
      }
      state.decoder = nil
    }

    private func contextState() -> sending LlamaContextState {
      let runtime = LlamaRuntime(model: self.model, parameters: self.contextParameters)
      return LlamaContextState(
        runtime: runtime,
        sequence: runtime.lease(copyingFrom: nil)!,
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.defaultSampling
      )
    }

    private func validate(_ context: LlamaContext) throws {
      guard context.storage.engineIdentity === self.identity else {
        throw EdgeToolsError.incompatibleContext
      }
    }

    private func generationState(
      from snapshot: LlamaContext.Snapshot
    ) -> ModelGenerationState {
      ModelGenerationState(
        contextState: snapshot.model,
        transcript: snapshot.transcript,
        reasoningEffort: snapshot.reasoningEffort,
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
    ) -> LlamaContext where Engine == LlamaEngine<Profile> {
      self.engine.context(
        transcript: transcript,
        reasoningEffort: reasoningEffort,
        tools: []
      )
    }

    public func context<Profile>(
      transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
      reasoningEffort: EdgeToolsReasoningEffort = .default,
      @EdgeToolsToolBuilder tools: () -> [any EdgeTool]
    ) -> LlamaContext where Engine == LlamaEngine<Profile> {
      self.engine.context(
        transcript: transcript,
        reasoningEffort: reasoningEffort,
        tools: tools()
      )
    }

    public func context<Profile>(
      systemPrompt: String,
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> LlamaContext where Engine == LlamaEngine<Profile> {
      self.engine.context(
        transcript: EdgeToolsTranscript(messages: [.system(systemPrompt)]),
        reasoningEffort: reasoningEffort,
        tools: []
      )
    }

    public func context<Profile>(
      systemPrompt: String,
      reasoningEffort: EdgeToolsReasoningEffort = .default,
      @EdgeToolsToolBuilder tools: () -> [any EdgeTool]
    ) -> LlamaContext where Engine == LlamaEngine<Profile> {
      self.engine.context(
        transcript: EdgeToolsTranscript(messages: [.system(systemPrompt)]),
        reasoningEffort: reasoningEffort,
        tools: tools()
      )
    }
  }

  // MARK: - XGrammar Cache

  #if XGrammar
    extension LlamaEngine where Profile.GrammarEngine == XGrammarEngine {
      public func clearCaches() {
        self.grammarEngine.clearCaches()
      }
    }

    extension EdgeToolsSession {
      public func clearCaches<Profile>()
      where Engine == LlamaEngine<Profile>, Profile.GrammarEngine == XGrammarEngine
      {
        self.engine.clearCaches()
      }
    }
  #endif

#endif
