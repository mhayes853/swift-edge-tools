#if Atomics
  import Atomics

  // MARK: - EdgeToolsModelPreparation

  public struct EdgeToolsModelPreparation: Sendable {
    public var metrics: EdgeToolsPrefillMetrics
    public var metadata: EdgeToolsMetadata

    public init(
      metrics: EdgeToolsPrefillMetrics,
      metadata: EdgeToolsMetadata = EdgeToolsMetadata()
    ) {
      self.metrics = metrics
      self.metadata = metadata
    }
  }

  // MARK: - EdgeToolsModelSample

  public struct EdgeToolsModelSample: Hashable, Sendable {
    public var tokenId: EdgeToolsToken.ID
    public var confidence: Float

    public init(tokenId: EdgeToolsToken.ID, confidence: Float) {
      self.tokenId = tokenId
      self.confidence = confidence
    }
  }

  // MARK: - EdgeToolsModelEngine

  public protocol EdgeToolsModelEngine: EdgeToolsEngine {
    associatedtype ModelGenerationState
    associatedtype GenerationParser: EdgeToolsGenerationParser
    associatedtype GrammarEngine: EdgeToolsGrammarEngine

    var tokenizer: any EdgeToolsTokenizer { get }
    var grammarEngine: GrammarEngine { get }

    func generationState(prompt: Prompt, context: Context) async throws -> ModelGenerationState

    func prepare(
      prompt: inout Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      parser: inout GenerationParser,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsModelPreparation

    func grammar(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      state: ModelGenerationState
    ) throws -> GrammarEngine.Grammar

    func decode(
      bitmask: GrammarBitmask,
      parameters: GenerateParameters,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsModelSample

    func stopTokenIds(state: ModelGenerationState) -> Set<EdgeToolsToken.ID>

    func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: Context
    ) async -> Result<EdgeToolsEngineGeneration, any Error>
  }

  extension EdgeToolsModelEngine {
    public func matcher(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      state: ModelGenerationState
    ) throws -> GrammarEngine.Matcher {
      try self.grammarEngine.matcher(
        for: self.grammar(
          prompt: prompt,
          tools: tools,
          parameters: parameters,
          state: state
        ),
        stopTokenIds: self.stopTokenIds(state: state)
      )
    }

    private func runGeneration(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel,
      state: inout ModelGenerationState,
      shouldStop: @escaping @Sendable () -> Bool
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !shouldStop() else { return .empty }

      let clock = ContinuousClock()
      var prompt = prompt
      var parser = GenerationParser()
      let generateStart = clock.now
      var preparation = try await self.prepare(
        prompt: &prompt,
        tools: tools,
        parameters: parameters,
        parser: &parser,
        state: &state
      )
      let stopTokenIds = self.stopTokenIds(state: state)
      var matcher = try self.matcher(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        state: state
      )
      var detokenizer = StreamingDetokenizer()
      var generatedTokens = [EdgeToolsToken]()
      var parts = [EdgeToolsGenerationPart]()
      var confidence = ConfidenceState()
      var durationToFirstToken: Duration?
      let maximumTokenCount = parameters.maxTokens ?? .max

      var bitmask: GrammarBitmask?
      if !matcher.isTerminated,
        !shouldStop(),
        generatedTokens.count < maximumTokenCount,
        generatedTokens.last.map({ !stopTokenIds.contains($0.id) }) ?? true
      {
        try Task.checkCancellation()
        bitmask = matcher.grammarBitmask()
      }

      while let currentBitmask = bitmask {
        let sample = try await self.decode(
          bitmask: currentBitmask,
          parameters: parameters,
          state: &state
        )
        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: clock.now)
        confidence.add(confidence: sample.confidence)

        let tokenString = detokenizer.decode(tokenId: sample.tokenId, using: self.tokenizer)
        let token = EdgeToolsToken(id: sample.tokenId, stringValue: tokenString)
        generatedTokens.append(token)
        guard matcher.accept(tokenId: token.id) else {
          throw EdgeToolsError.grammarRejectedToken(token: token)
        }

        channel.emit(token: token)
        let parsedParts = stopTokenIds.contains(token.id) ? [] : parser.accept(token: token)
        for part in parsedParts {
          parts.append(part)
          channel.emit(part: part)
        }

        bitmask = nil
        if !matcher.isTerminated,
          !shouldStop(),
          generatedTokens.count < maximumTokenCount,
          generatedTokens.last.map({ !stopTokenIds.contains($0.id) }) ?? true
        {
          try Task.checkCancellation()
          bitmask = matcher.grammarBitmask()
        }
      }

      for part in parser.finish() {
        parts.append(part)
        channel.emit(part: part)
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      preparation.metadata.generationConfidence = confidence.mean
      preparation.metadata.perTokenConfidences = confidence.perTokenConfidences
      var responseTokenIds = detokenizer.tokenIds
      if let lastTokenId = responseTokenIds.last, stopTokenIds.contains(lastTokenId) {
        responseTokenIds.removeLast()
      }
      let response = self.tokenizer.decode(tokens: responseTokenIds)
      let decodeDuration = generateStart.duration(to: clock.now) - finalDurationToFirstToken
      return EdgeToolsEngineGeneration(
        prefillMetrics: preparation.metrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: generatedTokens.count,
          duration: decodeDuration,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: shouldStop(),
        tokens: generatedTokens,
        response: response,
        parts: parts,
        metadata: preparation.metadata
      )
    }

    public func generate(
      prompt: Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: sending GenerateParameters,
      context: Context,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      self.generationTask(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          try await self.generationState(prompt: prompt, context: context)
        }
      )
    }

    func generationTask(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: sending GenerateParameters,
      context: Context,
      channel: sending EdgeToolsGenerationChannel,
      makeState: @escaping @Sendable () async throws -> ModelGenerationState
    ) -> AtomicGenerationTask {
      let taskState = ManagedAtomic(AtomicGenerationTask.State.queued.rawValue)

      // NB: The compiler's region isolation checker cannot express that these values are no
      // longer accessed after being transferred into the task, making these bindings safe.
      nonisolated(unsafe) let parameters = parameters
      nonisolated(unsafe) let channel = channel
      let task = Task {
        let didStart =
          taskState.compareExchange(
            expected: AtomicGenerationTask.State.queued.rawValue,
            desired: AtomicGenerationTask.State.running.rawValue,
            ordering: .relaxed
          )
          .exchanged
        guard didStart else { return EdgeToolsEngineGeneration.empty }

        var state = try await makeState()
        let result: Result<EdgeToolsEngineGeneration, any Error>
        do {
          result = .success(
            try await self.runGeneration(
              prompt: prompt,
              tools: tools,
              parameters: parameters,
              channel: channel,
              state: &state,
              shouldStop: {
                taskState.load(ordering: .relaxed) == AtomicGenerationTask.State.stopped.rawValue
              }
            )
          )
        } catch {
          result = .failure(error)
        }
        return try await self.finalize(state: state, result: result, context: context).get()
      }
      return AtomicGenerationTask(task: task, state: taskState)
    }
  }
#endif
