#if LlamaCore
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaModelProfile

  public protocol LlamaModelProfile: EdgeToolsModelProfile
  where GenerateParameters: LlamaGenerateParameters, Prompt == EdgeToolsTranscript {
    static func tokenIds(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      addGenerationPrompt: Bool
    ) throws -> [EdgeToolsToken.ID]
  }

  #if FoundationEssentials
    extension LlamaModelProfile {
      public static func tokenIds(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        addGenerationPrompt: Bool
      ) throws -> [EdgeToolsToken.ID] {
        guard let tokenizer = tokenizer as? any EdgeToolsChatTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        return try tokenizer.applyChatTemplate(
          messages: try prompt.chatTemplateMessages(),
          tools: tools.chatTemplateToolValues,
          addGenerationPrompt: addGenerationPrompt,
          additionalContext: Self.templateContext(prompt: prompt)
        )
        .map(\.id)
      }
    }
  #endif

  // MARK: - LlamaGenerateParameters

  public protocol LlamaGenerateParameters: EdgeToolsEngineGenerateParameters {
    var sampling: EdgeToolsFusedSamplingParameters { get }
  }

  #if XGrammar
    public struct DefaultLlamaGenerateParameters:
      LlamaGenerateParameters,
      EdgeToolsConstrainedGenerateParameters
    {
      public static var `default`: Self { Self() }

      public var sampling: EdgeToolsFusedSamplingParameters
      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?

      public init(
        sampling: EdgeToolsFusedSamplingParameters = EdgeToolsFusedSamplingParameters(),
        constraint: XGRGenerationConstraint = .unconstrained,
        maxTokens: Int? = 1024
      ) {
        self.sampling = sampling
        self.constraint = constraint
        self.maxTokens = maxTokens
      }
    }
  #endif

  // MARK: - LlamaContext

  public typealias LlamaContext<Profile> = EdgeToolsTranscriptContext<LlamaModelState<Profile>>
  where Profile: LlamaModelProfile

  // MARK: - LlamaModelHandle

  final class LlamaModelHandle: @unchecked Sendable {
    let api: LlamaApi
    let model: LlamaModelRef

    init(api: LlamaApi, model: LlamaModelRef) {
      self.api = api
      self.model = model
    }

    deinit {
      self.api.model.free(self.model)
    }
  }

  // MARK: - LlamaSequence

  /// One llama context holding the KV state of a fork family.
  ///
  /// All llama.cpp calls on the context go through the internal lock. Exclusive use during
  /// a generation is guaranteed by the transcript context's `isResponding` gate.
  final class LlamaSequence: @unchecked Sendable {
    private struct State {
      var context: LlamaContextRef?
      var cachedTokenIds = [EdgeToolsToken.ID]()
    }

    private static let decodeChunkSize = 512

    let handle: LlamaModelHandle
    let parameters: LlamaContextParameters
    private let state = Lock(State())

    private var api: LlamaApi {
      self.handle.api
    }

    init(handle: LlamaModelHandle, parameters: LlamaContextParameters) {
      self.handle = handle
      self.parameters = parameters
    }

    deinit {
      self.state.withBorrowedLock { state in
        if let context = state.context {
          self.handle.api.context.free(context)
        }
      }
    }

    /// Trims the KV cache to the common prefix and decodes the remaining suffix, leaving
    /// logits for the final token. Returns the number of decoded tokens.
    func prefill(tokenIds: [EdgeToolsToken.ID], wantsLogits: Bool) throws -> Int {
      try self.state.withLock { state in
        let context = try self.context(&state)
        let cached = state.cachedTokenIds
        var prefixCount = zip(cached, tokenIds).prefix { $0 == $1 }.count
        if wantsLogits && prefixCount == tokenIds.count && !tokenIds.isEmpty {
          prefixCount = tokenIds.count - 1
        }
        if prefixCount < cached.count {
          _ = self.api.context.memoryRemove(context, 0, prefixCount, -1)
        }
        state.cachedTokenIds = Array(tokenIds.prefix(prefixCount))
        var position = prefixCount
        while position < tokenIds.count {
          let chunk = Array(tokenIds[position..<min(position + Self.decodeChunkSize, tokenIds.count)])
          let isLast = position + chunk.count == tokenIds.count
          try self.api.context.decode(
            context,
            LlamaDecodeBatch(
              tokens: chunk,
              startPosition: position,
              sequenceId: 0,
              wantsLogits: wantsLogits && isLast
            )
          )
          state.cachedTokenIds.append(contentsOf: chunk)
          position += chunk.count
        }
        return tokenIds.count - prefixCount
      }
    }

    /// Decodes a single generated token at the end of the sequence.
    func append(tokenId: EdgeToolsToken.ID, wantsLogits: Bool) throws {
      try self.state.withLock { state in
        let context = try self.context(&state)
        try self.api.context.decode(
          context,
          LlamaDecodeBatch(
            tokens: [tokenId],
            startPosition: state.cachedTokenIds.count,
            sequenceId: 0,
            wantsLogits: wantsLogits
          )
        )
        state.cachedTokenIds.append(tokenId)
      }
    }

    func withLastLogits<R>(
      vocabularySize: Int,
      _ body: (UnsafeMutableBufferPointer<Float>) throws -> R
    ) throws -> R {
      try self.state.withLock { state in
        let context = try self.context(&state)
        guard let logits = self.api.context.lastLogits(context) else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The llama context has no logits for the last decoded token."
          )
        }
        return try body(UnsafeMutableBufferPointer(start: logits, count: vocabularySize))
      }
    }

    private func context(_ state: inout State) throws -> LlamaContextRef {
      if let context = state.context {
        return context
      }
      let context = try self.api.context.create(self.handle.model, self.parameters)
      state.context = context
      return context
    }
  }

  // MARK: - LlamaModelState

  public struct LlamaModelState<Profile: LlamaModelProfile> {
    private struct Generation {
      var pendingTokenId: EdgeToolsToken.ID?
      let sampler: EdgeToolsCPUFusedSampler
      var confidence = ConfidenceState()
    }

    private let sequence: LlamaSequence
    private let vocabularySizeValue: Int
    private let configuredSampling: EdgeToolsFusedSamplingParameters?
    private var generation: Generation?

    init(
      sequence: LlamaSequence,
      vocabularySize: Int,
      defaultSampling: EdgeToolsFusedSamplingParameters?
    ) {
      self.sequence = sequence
      self.vocabularySizeValue = vocabularySize
      self.configuredSampling = defaultSampling
    }

    public var vocabularySize: Int {
      self.vocabularySizeValue
    }

    public func forkedContextState(copyingCache: Bool) -> sending Self {
      // Forked user contexts receive a fresh llama context; copy-on-write forking through
      // sequence ids arrives with the fork-family phase. All shared state is the model
      // handle, which is immutable after engine initialization.
      let state = Self(
        sequence: LlamaSequence(
          handle: self.sequence.handle,
          parameters: self.sequence.parameters
        ),
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.configuredSampling
      )
      nonisolated(unsafe) let transferredState = state
      return transferredState
    }

    public func generationState() -> sending Self {
      // A generation continues on the same sequence; exclusivity is enforced by the
      // transcript context's `isResponding` gate, and the sequence serializes all llama
      // calls internally.
      nonisolated(unsafe) let state = Self(
        sequence: self.sequence,
        vocabularySize: self.vocabularySizeValue,
        defaultSampling: self.configuredSampling
      )
      return state
    }

    public mutating func prepare(
      prompt: inout EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser
    ) throws -> EdgeToolsGenerationLoop.Preparation {
      Profile.prepare(prompt: &prompt, tools: tools, parser: &parser)
      let tokenIds = try Profile.tokenIds(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        addGenerationPrompt: true
      )
      let clock = ContinuousClock()
      let start = clock.now
      let decodedCount = try self.sequence.prefill(tokenIds: tokenIds, wantsLogits: true)
      let defaultSampling =
        Profile.defaultSampling(prompt: prompt, parameters: parameters)
        ?? self.configuredSampling
        ?? EdgeToolsFusedSamplingParameters()
      self.generation = Generation(
        sampler: EdgeToolsCPUFusedSampler(
          parameters: parameters.sampling.applying(to: defaultSampling)
        )
      )
      return EdgeToolsGenerationLoop.Preparation(
        metrics: EdgeToolsPrefillMetrics(
          tokens: decodedCount,
          duration: start.duration(to: clock.now)
        )
      )
    }

    public mutating func prefill(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) throws -> EdgeToolsEnginePrefill {
      let tokenIds = try Profile.tokenIds(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        addGenerationPrompt: false
      )
      let clock = ContinuousClock()
      let start = clock.now
      let decodedCount = try self.sequence.prefill(tokenIds: tokenIds, wantsLogits: false)
      return EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(
          tokens: decodedCount,
          duration: start.duration(to: clock.now)
        )
      )
    }

    public func tokenIds(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) throws -> [EdgeToolsToken.ID] {
      try Profile.tokenIds(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        addGenerationPrompt: true
      )
    }

    public mutating func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters
    ) throws -> EdgeToolsToken.ID {
      guard var generation = self.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      if let pendingTokenId = generation.pendingTokenId {
        try self.sequence.append(tokenId: pendingTokenId, wantsLogits: true)
      }
      let sample = try self.sequence.withLastLogits(vocabularySize: self.vocabularySizeValue) {
        generation.sampler.sample(logits: $0, bitmask: bitmask)
      }
      generation.confidence.add(confidence: sample.confidence)
      generation.pendingTokenId = sample.tokenId
      self.generation = generation
      return sample.tokenId
    }

    public mutating func commitGeneration(stopTokenIds: Set<EdgeToolsToken.ID>) {
      guard let generation = self.generation else { return }
      if let pendingTokenId = generation.pendingTokenId,
        !stopTokenIds.contains(pendingTokenId)
      {
        try? self.sequence.append(tokenId: pendingTokenId, wantsLogits: false)
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
      return metadata
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

  // MARK: - LlamaEngine

  public final class LlamaEngine<Profile: LlamaModelProfile>:
    EdgeToolsEngine, EdgeToolsPrefillableEngine, EdgeToolsTokenizingEngine
  {
    public typealias Context = LlamaContext<Profile>
    public typealias ContextParameters = EdgeToolsTranscriptContextParameters
    public typealias Prompt = EdgeToolsTranscript.UserMessage
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias ModelGenerationState = LlamaGenerationState<Profile>

    private let handle: LlamaModelHandle
    private let contextParameters: LlamaContextParameters
    private let defaultSampling: EdgeToolsFusedSamplingParameters?
    private let vocabularySizeValue: Int
    private let identity = EdgeToolsEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
    public let tokenizer: any EdgeToolsTokenizer
    public let grammarEngine: Profile.GrammarEngine

    public init(
      api: LlamaApi,
      modelPath: String,
      modelParameters: LlamaModelParameters = LlamaModelParameters(),
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      api.backend.initialize()
      let model = try api.model.load(modelPath, modelParameters)
      self.handle = LlamaModelHandle(api: api, model: model)
      self.contextParameters = contextParameters
      self.defaultSampling = defaultSampling
      let tokenizer = LlamaTokenizer(api: api, model: model)
      self.tokenizer = tokenizer
      self.vocabularySizeValue = api.model.vocabularySize(model)
      var extraStopTokenIds = tokenizer.endOfGenerationTokenIds()
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
      let transcript = context.transcript(appending: prompt)
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
          self.generationState(from: try context.begin(appending: prompt))
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
        snapshot: context.begin(appending: promptPrefix),
        tools: tools,
        context: context
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
        var preparedPrompt = prompt
        _ = preparedPrompt
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
      nonisolated(unsafe) let state = LlamaModelState<Profile>(
        sequence: LlamaSequence(handle: self.handle, parameters: self.contextParameters),
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

  #if Llama && canImport(CLlama)
    extension LlamaEngine {
      public convenience init(
        modelPath: String,
        modelParameters: LlamaModelParameters = LlamaModelParameters(),
        contextParameters: LlamaContextParameters = LlamaContextParameters(),
        defaultSampling: EdgeToolsFusedSamplingParameters? = nil
      ) throws {
        try self.init(
          api: .vendored,
          modelPath: modelPath,
          modelParameters: modelParameters,
          contextParameters: contextParameters,
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
