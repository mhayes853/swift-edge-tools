#if Llama && canImport(CLlama)
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

  // MARK: - LlamaSequenceFamily

  /// One llama context whose KV cells are shared copy-on-write between the sequences of a
  /// fork family.
  ///
  /// All llama.cpp calls on the context go through the internal lock, so generations on
  /// different sequences of the family may interleave safely; exclusive use of one
  /// sequence during a generation is guaranteed by the transcript context's
  /// `isResponding` gate. Because the context has a single logits output, logits access
  /// is atomic with the decode that produced it, and a sequence re-decodes its final
  /// token when another sequence decoded in between.
  final class LlamaSequenceFamily: Sendable {
    /// Ownership of one sequence id; releasing it clears the sequence's KV cells.
    final class Lease: Sendable {
      let family: LlamaSequenceFamily
      let sequenceId: Int

      init(family: LlamaSequenceFamily, sequenceId: Int) {
        self.family = family
        self.sequenceId = sequenceId
      }

      deinit { self.family.release(sequenceId: self.sequenceId) }
    }

    private struct State: ~Copyable {
      var context: LlamaContextHandle?
      var cachedTokenIds = [Int: [EdgeToolsToken.ID]]()
      var allocatedSequenceIds = Set<Int>()
      var logitsSequenceId: Int?
    }

    private static let decodeChunkSize = 512

    let model: LlamaModel
    let parameters: LlamaContextParameters
    private let state = Lock(State())

    init(model: LlamaModel, parameters: LlamaContextParameters) {
      self.model = model
      self.parameters = parameters
    }

    /// Returns nil when the family has no sequence capacity left.
    func lease(copyingFrom parentSequenceId: Int?) -> Lease? {
      self.state.withLock { state in
        guard
          let sequenceId = (0..<self.parameters.maximumSequenceCount)
            .first(where: { !state.allocatedSequenceIds.contains($0) })
        else {
          return nil
        }
        state.allocatedSequenceIds.insert(sequenceId)
        if let parentSequenceId {
          state.context?.memoryCopy(
            source: parentSequenceId,
            destination: sequenceId,
            from: 0,
            to: -1
          )
          state.cachedTokenIds[sequenceId] = state.cachedTokenIds[parentSequenceId] ?? []
        } else {
          state.cachedTokenIds[sequenceId] = []
        }
        return Lease(family: self, sequenceId: sequenceId)
      }
    }

    /// Returns the number of tokens actually decoded after prefix reuse.
    func prefill(
      sequenceId: Int,
      tokenIds: [EdgeToolsToken.ID],
      wantsLogits: Bool
    ) throws -> Int {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        let cached = state.cachedTokenIds[sequenceId] ?? []
        var prefixCount = zip(cached, tokenIds).prefix { $0 == $1 }.count
        if wantsLogits && prefixCount == tokenIds.count && !tokenIds.isEmpty {
          prefixCount = tokenIds.count - 1
        }
        prefixCount = self.trim(sequenceId: sequenceId, to: prefixCount, state: &state)
        try self.decode(
          tokenIds: Array(tokenIds[prefixCount...]),
          sequenceId: sequenceId,
          wantsLogits: wantsLogits,
          state: &state
        )
        return tokenIds.count - prefixCount
      }
    }

    /// Regenerates the logits first when another sequence decoded since they were produced.
    func withCurrentLogits<R>(
      sequenceId: Int,
      appending pendingTokenId: EdgeToolsToken.ID?,
      vocabularySize: Int,
      _ body: (inout MutableSpan<Float>) throws -> R
    ) throws -> R {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        if let pendingTokenId {
          try self.decode(
            tokenIds: [pendingTokenId],
            sequenceId: sequenceId,
            wantsLogits: true,
            state: &state
          )
        } else if state.logitsSequenceId != sequenceId {
          let cached = state.cachedTokenIds[sequenceId] ?? []
          guard !cached.isEmpty else {
            throw LlamaRuntimeError(
              code: .decodeFailed,
              message: "The sequence has no decoded tokens to produce logits from."
            )
          }
          let prefixCount = self.trim(sequenceId: sequenceId, to: cached.count - 1, state: &state)
          try self.decode(
            tokenIds: Array(cached[prefixCount...]),
            sequenceId: sequenceId,
            wantsLogits: true,
            state: &state
          )
        }
        guard let logits = state.context?.lastLogits() ?? nil else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The llama context has no logits for the last decoded token."
          )
        }
        var span = MutableSpan<Float>(
          _unsafeElements: UnsafeMutableBufferPointer(start: logits, count: vocabularySize)
        )
        return try body(&span)
      }
    }

    func append(tokenId: EdgeToolsToken.ID, sequenceId: Int) throws {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        try self.decode(
          tokenIds: [tokenId],
          sequenceId: sequenceId,
          wantsLogits: false,
          state: &state
        )
      }
    }

    private func release(sequenceId: Int) {
      self.state.withLock { state in
        _ = state.context?.memoryRemove(sequenceId: sequenceId, from: 0, to: -1)
        state.cachedTokenIds[sequenceId] = nil
        state.allocatedSequenceIds.remove(sequenceId)
        if state.logitsSequenceId == sequenceId {
          state.logitsSequenceId = nil
        }
      }
    }

    /// Drops the cells the sequence holds past `prefixCount`, returning the token count it
    /// actually retains. Recurrent memory modules only support clearing a sequence whole, so
    /// a rejected removal wipes the sequence and the caller decodes from scratch.
    private func trim(sequenceId: Int, to prefixCount: Int, state: inout State) -> Int {
      let cached = state.cachedTokenIds[sequenceId] ?? []
      guard prefixCount < cached.count else { return prefixCount }
      var prefixCount = prefixCount
      if state.context?.memoryRemove(sequenceId: sequenceId, from: prefixCount, to: -1) != true {
        _ = state.context?.memoryRemove(sequenceId: sequenceId, from: 0, to: -1)
        prefixCount = 0
      }
      state.cachedTokenIds[sequenceId] = Array(cached.prefix(prefixCount))
      return prefixCount
    }

    private func decode(
      tokenIds: [EdgeToolsToken.ID],
      sequenceId: Int,
      wantsLogits: Bool,
      state: inout State
    ) throws {
      guard state.context != nil else {
        throw LlamaRuntimeError(
          code: .contextCreationFailed,
          message: "The llama context is unavailable."
        )
      }
      for start in stride(from: 0, to: tokenIds.count, by: Self.decodeChunkSize) {
        let end = min(start + Self.decodeChunkSize, tokenIds.count)
        let chunk = Array(tokenIds[start..<end])
        try state.context?.decode(
          tokenIds: chunk,
          startPosition: state.cachedTokenIds[sequenceId]?.count ?? 0,
          sequenceId: sequenceId,
          wantsLogits: wantsLogits && end == tokenIds.count
        )
        state.cachedTokenIds[sequenceId, default: []].append(contentsOf: chunk)
      }
      if wantsLogits {
        state.logitsSequenceId = sequenceId
      }
    }

    private func ensureContext(_ state: inout State) throws {
      guard state.context == nil else { return }
      state.context = try self.model.createContext(parameters: self.parameters)
    }
  }

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
      // Exhausting the family's sequence capacity falls back to a fresh, cache-cold family.
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
          parameters: self.family.parameters
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
      // Generations continue on the context's own sequence: forking one here would hand it
      // a cold cache and sever KV continuity. Exclusivity comes from `isResponding`.
      nonisolated(unsafe) let state = Self(
        family: self.family,
        lease: self.lease,
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
      let decodedCount = try self.family.prefill(
        sequenceId: self.lease.sequenceId,
        tokenIds: tokenIds,
        wantsLogits: true
      )
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
      let decodedCount = try self.family.prefill(
        sequenceId: self.lease.sequenceId,
        tokenIds: tokenIds,
        wantsLogits: false
      )
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
      let sample = try self.family.withCurrentLogits(
        sequenceId: self.lease.sequenceId,
        appending: generation.pendingTokenId,
        vocabularySize: self.vocabularySizeValue
      ) {
        generation.sampler.sample(logits: &$0, bitmask: bitmask)
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

    private let model: LlamaModel
    private let contextParameters: LlamaContextParameters
    private let defaultSampling: EdgeToolsFusedSamplingParameters?
    private let vocabularySizeValue: Int
    private let identity = EdgeToolsEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
    public let tokenizer: any EdgeToolsTokenizer
    public let grammarEngine: Profile.GrammarEngine

    public init(
      modelPath: String,
      modelParameters: LlamaModelParameters = LlamaModelParameters(),
      contextParameters: LlamaContextParameters = LlamaContextParameters(),
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      let model = try LlamaModel(path: modelPath, parameters: modelParameters)
      self.model = model
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
        parameters: self.contextParameters
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
