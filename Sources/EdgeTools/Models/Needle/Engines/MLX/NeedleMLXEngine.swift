#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Atomics

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: EdgeToolsEngine {
    public typealias Prompt = NeedlePrompt

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

    private struct State: ~Copyable {
      let grammarEngine: XGrammarCompiler
      let model: NeedleMLXModel
      let matcherPool: XGrammarToolCallMatcherPool
    }

    private let state: Lock<State>
    private let tokenizer: Lock<any EdgeToolsTokenizer>

    private let clock = ContinuousClock()

    public convenience init(
      from url: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: url, isNeedleModel: true)
      let lockedTokenizer = Lock(consume tokenizer)
      let model = try loadNeedleMLXModel(
        from: url,
        editConfiguration: editConfiguration
      )
      let grammarEngine = lockedTokenizer.withLock {
        try? XGrammarCompiler(tokenizerInfo: try XGrammarTokenizerInfo.needle(erasedTokenizer: $0))
      }
      guard let grammarEngine else {
        throw NeedleMLXEngineError.failedToLoadGrammarEngine
      }
      self.init(
        tokenizer: consume lockedTokenizer,
        model: model,
        grammarEngine: consume grammarEngine
      )
    }

    public init<Tokenizer: EdgeToolsTokenizer>(
      tokenizer: consuming sending Tokenizer,
      model: sending NeedleMLXModel,
      grammarEngine: consuming sending XGrammarCompiler
    ) {
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: XGrammarToolCallMatcherPool.needle()
        )
      )
      self.tokenizer = Lock(consume tokenizer)
    }

    private init(
      tokenizer: consuming sending Lock<any EdgeToolsTokenizer>,
      model: sending NeedleMLXModel,
      grammarEngine: consuming sending XGrammarCompiler
    ) {
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: XGrammarToolCallMatcherPool.needle()
        )
      )
      self.tokenizer = consume tokenizer
    }

    public func tokenize(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      try self.tokenizer.withBorrowedLock { try prompt.tokenized(tools: tools, using: $0) }
    }

    public func clearCaches() {
      self.state.withLock {
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: NeedlePrompt,
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
      prompt: NeedlePrompt,
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
      state.model.reset()
      defer { state.model.reset() }

      let generationStartSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let generateStart = self.clock.now

      let sampler = parameters.sampler
      var processor = parameters.processor
      var cache = state.model.newCache(parameters: nil)
      let (prefillOutput, prefillMetrics, postPrefillSnapshot) = try self.prefill(
        prompt: prompt,
        tools: tools,
        model: state.model,
        cache: cache,
        processor: &processor,
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      try Task.checkCancellation()
      guard var output = prefillOutput else { preconditionFailure("Model received empty input.") }
      var durationToFirstToken: Duration?

      var detokenizer = StreamingDetokenizer()
      var parser = NeedleToolCallParser()
      var generatedTokens = [EdgeToolsToken]()
      var confidence = EdgeToolsConfidenceState()
      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.withBorrowedLock({ $0.eosTokenId })
      {
        try Task.checkCancellation()
        let processedLogits = processor?.process(logits: output.logits) ?? output.logits
        let logits = applyBitmaskMLX(
          logits: processedLogits[0..., -1, 0...],
          mask: matcher.bitmask()
        )
        confidence.addMLX(logits: logits)

        let token = sampler.sample(logits: logits)
        let tokenId = token.item(EdgeToolsToken.ID.self)

        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)
        let tokenString = self.tokenizer.withBorrowedLock {
          detokenizer.decode(tokenId: tokenId, using: $0)
        }
        let needleToken = EdgeToolsToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(needleToken)
        guard matcher.accept(tokenId: needleToken.id) else {
          throw NeedleMLXEngineError.grammarRejectedToken(token: needleToken)
        }
        let rawToolCall = parser.accept(token: needleToken)
        channel.emit(token: needleToken)
        if let rawToolCall {
          channel.emit(toolCall: rawToolCall)
        }
        processor?.didSample(token: token)

        maybeQuantizeKVCache(
          cache: &cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize
        )

        let inputText = LMInput.Text(tokens: token)
        output = state.model(inputText[text: .newAxis], cache: cache, state: output.state)
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      let postDecodeSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      var metadata = EdgeToolsMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = postPrefillSnapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = postDecodeSnapshot
      metadata.generationConfidence = confidence.mean
      metadata.perTokenConfidences = confidence.perTokenConfidences
      return EdgeToolsEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: generatedTokens.count,
          duration: generateStart.duration(to: self.clock.now) - finalDurationToFirstToken,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        metadata: metadata
      )
    }

    private func prefill(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      model: NeedleMLXModel,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> (LMOutput?, EdgeToolsPrefillMetrics, Memory.Snapshot) {
      let tokens = try self.tokenizer.withBorrowedLock {
        try $0.encode(text: prompt.formatted(tools: tools))
      }
      let input = LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
      guard input.text.tokens.size <= model.configuration.encoderMaxLength else {
        throw NeedleMLXEngineError.contextLengthExceeded(
          tokens: input.text.tokens.size,
          maximum: model.configuration.encoderMaxLength
        )
      }

      let prefillStart = self.clock.now
      processor?.prompt(input.text.tokens)
      let output = try model.prepare(input.text.tokens, cache: cache, windowSize: nil)
      let metrics = EdgeToolsPrefillMetrics(
        tokens: input.text.tokens.size,
        duration: prefillStart.duration(to: self.clock.now)
      )
      let snapshot = Memory.synchronizedSnapshot(synchronize: synchronize)
      return (output, metrics, snapshot)
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

  // MARK: - NeedleMLXEngineError

  public struct NeedleMLXEngineError: Hashable, Error {
    public let message: String

    public static let failedToLoadConfiguration = Self(
      message: "Could not load model configuration."
    )

    public static let failedToLoadGrammarEngine = Self(
      message: "Could not load grammar engine."
    )

    public static func grammarRejectedToken(token: EdgeToolsToken) -> Self {
      Self(
        message:
          "Token (ID=\(token.id), VALUE=\(token.stringValue)) was rejected by the grammar matcher."
      )
    }

    public static func contextLengthExceeded(tokens: Int, maximum: Int) -> Self {
      Self(message: "Prompt token count (\(tokens)) exceeds the model context length (\(maximum)).")
    }
  }

  // MARK: - Helpers

  package func loadNeedleMLXModel(
    from url: URL,
    editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
  ) throws -> sending NeedleMLXModel {
    guard var configuration = try NeedleModelConfiguration.decode(in: url) else {
      throw NeedleMLXEngineError.failedToLoadConfiguration
    }
    editConfiguration(&configuration)
    let model = NeedleMLXModel(configuration: configuration)
    try model.loadWeights(from: url.appending(path: "model.safetensors"))
    return model
  }
#endif
