#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Atomics

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: EdgeToolEngine {
    public struct GenerateParameters: EdgeToolEngineGenerateParameters {
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
      let matcherPool: NeedleGrammarMatcherPool
    }

    private let state: Lock<State>
    private let tokenizer: Lock<any EdgeToolsTokenizer & ~Copyable>

    private let clock = ContinuousClock()

    public convenience init(
      from url: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
    ) throws {
      let tokenizer = try Self.loadTokenizer(from: url.appending(path: "tokenizer.model"))
      let model = try loadNeedleMLXModel(
        from: url,
        editConfiguration: editConfiguration
      )
      let grammarEngine = tokenizer.withLock { XGrammarCompiler.needle(erasedTokenizer: $0) }
      guard let grammarEngine else {
        throw XGrammarError(message: "Needle requires a tokenizer with an EOS token.")
      }
      self.init(tokenizer: tokenizer, model: model, grammarEngine: consume grammarEngine)
    }

    public init<Tokenizer: EdgeToolsTokenizer & ~Copyable>(
      tokenizer: consuming sending Tokenizer,
      model: sending NeedleMLXModel,
      grammarEngine: consuming sending XGrammarCompiler
    ) {
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: NeedleGrammarMatcherPool()
        )
      )
      self.tokenizer = Lock(consume tokenizer)
    }

    private init(
      tokenizer: consuming sending Lock<any EdgeToolsTokenizer & ~Copyable>,
      model: sending NeedleMLXModel,
      grammarEngine: consuming sending XGrammarCompiler
    ) {
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: NeedleGrammarMatcherPool()
        )
      )
      self.tokenizer = consume tokenizer
    }

    private static func loadTokenizer(
      from modelURL: URL
    ) throws -> Lock<any EdgeToolsTokenizer & ~Copyable> {
      let tokenizer = try EdgeToolsSPTokenizer(modelURL: modelURL)
      return Lock(consume tokenizer)
    }

    public func tokenize(prompt: EdgeToolsPrompt) async throws -> [EdgeToolsToken] {
      self.tokenizer.withBorrowedLock { prompt.needleTokenized(using: $0) }
    }

    public func clearCaches() {
      self.state.withLock {
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: EdgeToolsPrompt,
      parameters: GenerateParameters,
      onToken: @escaping @Sendable (EdgeToolsToken, EdgeRawToolCall?) -> Void
    ) throws -> some EdgeToolEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        try self.state.withLock { state in
          try self.generate(
            prompt: prompt,
            parameters: parameters,
            onToken: onToken,
            state: &state,
            isStopped: isStopped
          )
        }
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: EdgeToolsPrompt,
      parameters: GenerateParameters,
      onToken: @escaping @Sendable (EdgeToolsToken, EdgeRawToolCall?) -> Void,
      state: inout sending State,
      isStopped: ManagedAtomic<Bool>
    ) throws -> EdgeToolEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = try state.matcherPool.matcher(
        tools: prompt.tools,
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
        let tokenString = self.tokenizer.withBorrowedLock { detokenizer.decode(tokenId: tokenId, using: $0)
                 }
        let needleToken = EdgeToolsToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(needleToken)
        guard matcher.accept(tokenId: needleToken.id) else {
          throw NeedleMLXEngineError.grammarRejectedToken(token: needleToken)
        }
        onToken(needleToken, parser.accept(token: needleToken))
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
      return EdgeToolEngineGeneration(
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
      prompt: EdgeToolsPrompt,
      model: NeedleMLXModel,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> (LMOutput?, EdgeToolsPrefillMetrics, Memory.Snapshot) {
      let tokens = self.tokenizer.withBorrowedLock {
        $0.encode(text: prompt.needleFormatted())
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
