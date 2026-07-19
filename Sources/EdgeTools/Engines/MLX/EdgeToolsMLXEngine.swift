#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLLM
  import MLXLMCommon
  import Foundation
  import Atomics

  // MARK: - EdgeToolsMLXEngine

  public final class EdgeToolsMLXEngine<Model: EdgeToolsLanguageModel>: EdgeToolsEngine {
    public typealias Prompt = Model.Prompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self {
        Self()
      }

      private var _sampler: @Sendable () -> any LogitSampler
      private var _processor: @Sendable () -> (any LogitProcessor)?

      public var sampler: any LogitSampler {
        self._sampler()
      }

      public var processor: (any LogitProcessor)? {
        self._processor()
      }

      public var toolCallRange: GrammarToolCallRange
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any LogitSampler = ArgMaxSampler(),
        processor: @autoclosure @escaping @Sendable () -> (any LogitProcessor)? = nil,
        toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self._sampler = sampler
        self._processor = processor
        self.toolCallRange = toolCallRange
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
        self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
      }
    }

    private struct CachedPrefill {
      let tokenIDs: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
    }

    private struct GenerationPreparation {
      let output: LMOutput
      let cache: [any KVCache]
      let metrics: EdgeToolsPrefillMetrics
      let snapshot: Memory.Snapshot
    }

    private struct State: ~Copyable {
      let grammarEngine: XGRCompiler
      let model: Model
      let matcherPool: XGRToolCallMatcherPool
      var cachedPrefill: CachedPrefill?
    }

    private let state: Lock<State>
    private let tokenizer: any EdgeToolsTokenizer
    private let clock = ContinuousClock()

    public init(
      model: sending Model,
      tokenizer: sending any EdgeToolsTokenizer
    ) throws {
      self.state = try Lock {
        let grammarEngine = try model.grammarCompiler(using: tokenizer)
        return State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: XGRToolCallMatcherPool(makeGrammar: model.grammar),
          cachedPrefill: nil
        )
      }
      self.tokenizer = tokenizer
    }

    private init(
      loadingFrom directoryURL: URL,
      tokenizer: sending any EdgeToolsTokenizer,
      model: (Model.ModelConfiguration) throws -> Model
    ) throws {
      self.state = try Lock {
        let model = try loadEdgeToolsLanguageModel(Model.self, from: directoryURL, model: model)
        let grammarEngine = try model.grammarCompiler(using: tokenizer)
        return State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: XGRToolCallMatcherPool(makeGrammar: model.grammar),
          cachedPrefill: nil
        )
      }
      self.tokenizer = tokenizer
    }

    public convenience init(
      from directoryURL: URL,
      model: (Model.ModelConfiguration) throws -> Model
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: directoryURL)
      try self.init(
        loadingFrom: directoryURL,
        tokenizer: tokenizer,
        model: model
      )
    }

    public func tokenize(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      try self.state.withBorrowedLock { state in
        let input = try state.model.process(
          prompt: prompt,
          tools: tools,
          using: self.tokenizer
        )
        let tokenIDs = input.text.tokens.asArray(EdgeToolsToken.ID.self)
        let tokens = self.tokenizer.convertIdsToTokens(tokenIDs)
        return zip(tokenIDs, tokens)
          .compactMap { tokenID, token in
            token.map { EdgeToolsToken(id: tokenID, stringValue: $0) }
          }
      }
    }

    public func clearCaches() {
      self.state.withLock {
        $0.cachedPrefill = nil
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        try self.state.withLock { state in
          try self.generate(
            prompt: prompt,
            tools: tools,
            parameters: parameters,
            channel: channel,
            state: &state,
            isStopped: isStopped
          )
        }
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel,
      state: inout sending State,
      isStopped: ManagedAtomic<Bool>
    ) throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = try state.matcherPool.matcher(
        tools: tools,
        range: parameters.toolCallRange,
        compilingWith: state.grammarEngine
      )
      matcher.reset()

      let generationStartSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let generateStart = self.clock.now
      let input = try state.model.process(
        prompt: prompt,
        tools: tools,
        using: self.tokenizer
      )
      let sampler = parameters.sampler
      var processor = parameters.processor
      let preparation = try self.prepareForGeneration(
        input: input,
        state: &state,
        processor: &processor,
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      var output = preparation.output
      var cache = preparation.cache
      try Task.checkCancellation()
      var durationToFirstToken: Duration?

      var detokenizer = StreamingDetokenizer()
      var parser = Model.ToolCallParser()
      var generatedTokens = [EdgeToolsToken]()
      var confidence = EdgeToolsConfidenceState()
      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        let processedLogits = processor?.process(logits: output.logits) ?? output.logits
        let logits = applyBitmaskMLX(
          logits: processedLogits[0..., -1, 0...],
          mask: matcher.bitmask()
        )
        confidence.addMLX(logits: logits)

        let sampledToken = sampler.sample(logits: logits)
        let tokenID = sampledToken.item(EdgeToolsToken.ID.self)

        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)
        let tokenString = detokenizer.decode(tokenId: tokenID, using: self.tokenizer)
        let token = EdgeToolsToken(id: tokenID, stringValue: tokenString)
        generatedTokens.append(token)
        guard matcher.accept(tokenId: token.id) else {
          throw EdgeToolsMLXEngineError.grammarRejectedToken(token: token)
        }
        let rawToolCall = parser.accept(token: token)
        channel.emit(token: token)
        if let rawToolCall {
          channel.emit(toolCall: rawToolCall)
        }
        processor?.didSample(token: sampledToken)

        maybeQuantizeKVCache(
          cache: &cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize
        )

        let inputText = LMInput.Text(tokens: sampledToken)
        output = state.model(
          inputText[text: .newAxis],
          cache: cache,
          state: output.state
        )
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      let postDecodeSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      var metadata = EdgeToolsMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = preparation.snapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = postDecodeSnapshot
      metadata.generationConfidence = confidence.mean
      metadata.perTokenConfidences = confidence.perTokenConfidences
      let response = self.tokenizer.decode(tokens: detokenizer.tokenIds)
      return EdgeToolsEngineGeneration(
        prefillMetrics: preparation.metrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: generatedTokens.count,
          duration: generateStart.duration(to: self.clock.now) - finalDurationToFirstToken,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        response: response,
        metadata: metadata
      )
    }

    private func prepareForGeneration(
      input: LMInput,
      state: inout State,
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> GenerationPreparation {
      let tokenIDs = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      if let cachedPrefill = state.cachedPrefill,
        tokenIDs.starts(with: cachedPrefill.tokenIDs)
      {
        let suffixCount = tokenIDs.count - cachedPrefill.tokenIDs.count
        let prefillStart = self.clock.now
        processor?.prompt(input.text.tokens)
        let cache = cachedPrefill.cache.map { $0.copy() }
        let output =
          if suffixCount == 0 {
            cachedPrefill.output
          } else {
            state.model(
              input.text[cachedPrefill.tokenIDs.count...][text: .newAxis],
              cache: cache,
              state: cachedPrefill.output.state
            )
          }
        let metrics = EdgeToolsPrefillMetrics(
          tokens: suffixCount,
          duration: prefillStart.duration(to: self.clock.now)
        )
        let snapshot = Memory.synchronizedSnapshot(synchronize: synchronize)
        return GenerationPreparation(
          output: output,
          cache: cache,
          metrics: metrics,
          snapshot: snapshot
        )
      }

      let cache = state.model.newCache(parameters: nil)
      let prefillStart = self.clock.now
      let output = try self.prepare(
        input: input,
        model: state.model,
        cache: cache,
        processor: &processor
      )
      let metrics = EdgeToolsPrefillMetrics(
        tokens: tokenIDs.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      let snapshot = Memory.synchronizedSnapshot(synchronize: synchronize)
      return GenerationPreparation(
        output: output,
        cache: cache,
        metrics: metrics,
        snapshot: snapshot
      )
    }

    private func prepare(
      input: LMInput,
      model: Model,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?
    ) throws -> LMOutput {
      processor?.prompt(input.text.tokens)
      switch try model.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else { throw EdgeToolsMLXEngineError.emptyInput }
        return model(
          tokens[text: .newAxis],
          cache: cache.isEmpty ? nil : cache,
          state: nil
        )
      }
    }
  }

  // MARK: - EdgeToolsPrefillableEngine

  extension EdgeToolsMLXEngine: EdgeToolsPrefillableEngine where Model: LLMModel {
    public func prefill(
      promptPrefix: Model.Prompt,
      tools: [EdgeToolDefinition]
    ) async throws -> EdgeToolsEnginePrefill {
      try self.state.withLock { state in
        try Task.checkCancellation()
        let input = try state.model.process(
          prompt: promptPrefix,
          tools: tools,
          using: self.tokenizer
        )
        return try self.prefill(input: input, state: &state)
      }
    }

    private func prefill(
      input: consuming sending LMInput,
      state: inout sending State
    ) throws -> EdgeToolsEnginePrefill {
      try Task.checkCancellation()
      let tokenIDs = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let cache = state.model.newCache(parameters: nil)
      var processor: (any LogitProcessor)?
      let prefillStart = self.clock.now
      let output = try self.prepare(
        input: input,
        model: state.model,
        cache: cache,
        processor: &processor
      )
      eval(output.logits)
      eval(cache)
      state.cachedPrefill = CachedPrefill(
        tokenIDs: tokenIDs,
        cache: cache.map { $0.copy() },
        output: output
      )
      let snapshot = Memory.synchronizedSnapshot(synchronize: true)
      var metadata = EdgeToolsMetadata()
      metadata.mlxEnginePostPrefillMemorySnapshot = snapshot
      return EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(
          tokens: tokenIDs.count,
          duration: prefillStart.duration(to: self.clock.now)
        ),
        metadata: metadata
      )
    }
  }

  // MARK: - Synchronized Memory Snapshot

  extension Memory {
    fileprivate static func synchronizedSnapshot(synchronize: Bool) -> Snapshot {
      if synchronize {
        Stream.defaultStream(.defaultDevice()).synchronize()
      }
      return Self.snapshot()
    }
  }

  // MARK: - EdgeToolsMLXEngineError

  public struct EdgeToolsMLXEngineError: Hashable, Error {
    public let message: String

    public static let emptyInput = Self(message: "Model received empty input.")
    public static let failedToLoadConfiguration = Self(
      message: "Could not load model configuration."
    )
    public static let failedToLoadWeights = Self(message: "Could not load model weights.")
    public static let unsupportedTokenizer = Self(
      message: "The model does not support the provided tokenizer."
    )

    public static func grammarRejectedToken(token: EdgeToolsToken) -> Self {
      Self(
        message:
          "Token (ID=\(token.id), VALUE=\(token.stringValue)) was rejected by the grammar matcher."
      )
    }
  }
#endif
