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
    public typealias Prompt = EdgeToolsTranscript.UserMessage
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias ModelGenerationState = LlamaGenerationState<Profile>

    private let model: LlamaModelBox
    private let multimodalProjector: LlamaMultimodalProjector?
    private let contextParameters: LlamaContextParameters
    private let defaultSampling: EdgeToolsFusedSamplingParameters?
    private let vocabularySizeValue: Int
    private let identity = EdgeToolsEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
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

    /// Loads a text model and its matching multimodal projector.
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
      self.multimodalProjector = try projectorPath.map {
        try LlamaMultimodalProjector(path: $0, model: model, parameters: multimodalParameters)
      }
      self.contextParameters = contextParameters
      self.defaultSampling = defaultSampling
      let tokenizer = LlamaTokenizer(model: model)
      self.tokenizer = tokenizer
      self.vocabularySizeValue = tokenizer.vocabularySize
      var extraStopTokenIds = tokenizer.endOfGenerationTokenIds()
      extraStopTokenIds.formUnion(
        Profile.extraStopTokens.compactMap { tokenizer.token(forText: $0)?.id }
      )
      var grammarStopTokenIds = extraStopTokenIds
      if let eosTokenId = tokenizer.eos?.id {
        extraStopTokenIds.remove(eosTokenId)
        grammarStopTokenIds.insert(eosTokenId)
      }
      self.grammarEngine = try Profile.grammarEngine(
        tokenizer: tokenizer,
        vocabularySize: self.vocabularySizeValue,
        stopTokenIds: grammarStopTokenIds
      )
      self.generationLoop = EdgeToolsGenerationLoop(
        tokenizer: tokenizer,
        extraStopTokenIds: extraStopTokenIds
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
        model: self.makeModelState(),
        engineIdentity: self.identity
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: LlamaContext<Profile>
    ) async throws -> [EdgeToolsToken] {
      try self.validate(context)
      let transcript = context.transcript(appending: .user(prompt))
      let model = self.makeModelState()
      let tokenIds = try model.tokenIds(
        prompt: transcript,
        tools: tools,
        tokenizer: self.tokenizer
      )
      return self.tokenizer.tokens(forIds: tokenIds).compactMap { $0 }
    }

    public func generate(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: LlamaContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          self.generationState(from: try context.begin(appending: .user(prompt)))
        }
      )
    }

    public func generate(
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: LlamaContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        prompt: .user(""),
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          self.generationState(from: try context.begin())
        }
      )
    }

    public func prefill(
      promptPrefix: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: LlamaContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try self.prefill(
        snapshot: context.begin(appending: .user(promptPrefix)),
        tools: tools,
        context: context
      )
    }

    /// Pays the setup a first generation would otherwise absorb into its time to first token:
    /// allocating the llama context, and parsing the chat template for the given tools.
    public func warmUp(tools: [EdgeToolDefinition] = [], context: LlamaContext<Profile>) throws {
      try self.validate(context)
      let snapshot = try context.begin()
      nonisolated(unsafe) let model = snapshot.model
      defer {
        context.finish(generation: nil, revision: snapshot.revision, model: model)
      }
      try model.warmUp()
      _ = try model.tokenIds(
        prompt: snapshot.transcript,
        tools: tools,
        tokenizer: self.tokenizer
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
      prompt: EdgeToolsTranscript.UserMessage,
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
              grammar: { state in
                try Profile.grammar(
                  prompt: state.transcript,
                  tools: tools,
                  parameters: parameters,
                  grammarEngine: self.grammarEngine
                )
              },
              prepare: { parser, state in
                try state.model.prepare(
                  prompt: &state.transcript,
                  tools: tools,
                  tokenizer: self.tokenizer,
                  parameters: parameters,
                  parser: &parser
                )
              },
              decode: { bitmask, state in
                try state.model.decode(bitmask: bitmask, parameters: parameters)
              }
            )
          )
        } catch {
          result = .failure(error)
        }
        return try self.finalize(state: state, result: result, context: context).get()
      }
    }

    private func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: LlamaContext<Profile>
    ) -> Result<EdgeToolsEngineGeneration, any Error> {
      var state = state
      let metadata = state.model.finish()

      let generation: EdgeToolsEngineGeneration?
      let finalResult: Result<EdgeToolsEngineGeneration, any Error>
      switch result {
      case .success(var value):
        state.model.commitGeneration(stopTokenIds: self.generationLoop.stopTokenIds)
        value.metadata.merge(metadata) { _, finalValue in finalValue }
        generation = value
        finalResult = .success(value)
      case .failure(let error):
        state.model.resetGeneration()
        generation = nil
        finalResult = .failure(error)
      }
      // The state is returned exactly once and is not accessed after this point. Region
      // isolation does not currently infer that exclusivity through the synchronous
      // context method.
      nonisolated(unsafe) let restoredModel = state.model
      context.finish(
        generation: generation,
        revision: state.revision,
        model: restoredModel
      )
      return finalResult
    }

    private func prefill(
      snapshot: LlamaContext<Profile>.Snapshot,
      tools: [EdgeToolDefinition],
      context: LlamaContext<Profile>
    ) throws -> EdgeToolsEnginePrefill {
      var model = snapshot.model
      do {
        let prefill = try model.prefill(
          prompt: snapshot.transcript,
          tools: tools,
          tokenizer: self.tokenizer
        )
        nonisolated(unsafe) let restoredModel = model
        context.finish(generation: nil, revision: snapshot.revision, model: restoredModel)
        return prefill
      } catch {
        model.resetGeneration()
        nonisolated(unsafe) let restoredModel = model
        context.finish(generation: nil, revision: snapshot.revision, model: restoredModel)
        throw error
      }
    }

    private func makeModelState() -> sending LlamaModelState<Profile> {
      let family = LlamaSequenceFamily(
        model: self.model,
        parameters: self.contextParameters,
        multimodalProjector: self.multimodalProjector
      )
      nonisolated(unsafe) let state = LlamaModelState<Profile>(
        family: family,
        lease: family.lease(copyingFrom: nil)!,
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.defaultSampling
      )
      return state
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
        model: snapshot.model,
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
