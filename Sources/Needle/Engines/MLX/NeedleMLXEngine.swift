#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Tokenizers
  import Atomics

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine {
    public final class GenerationTask: NeedleEngineGenerationTask {
      private let task: Task<NeedleEngineGeneration, any Error>
      private let isStopped: ManagedAtomic<Bool>

      fileprivate init(
        task: sending Task<NeedleEngineGeneration, any Error>,
        isStopped: ManagedAtomic<Bool>
      ) {
        self.task = task
        self.isStopped = isStopped
      }

      public var value: NeedleEngineGeneration {
        get async throws {
          try await withTaskCancellationHandler {
            try await self.task.value
          } onCancel: {
            self.task.cancel()
          }
        }
      }

      public func stop() {
        self.isStopped.store(true, ordering: .relaxed)
      }
    }

    public struct GenerateParameters: NeedleEngineGenerateParameters {
      public static var `default`: Self {
        Self()
      }

      public var sampler: LogitSampler
      public var processor: LogitProcessor?
      public var toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: any LogitSampler = ArgMaxSampler(),
        processor: (any LogitProcessor)? = nil,
        toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange =
          .unbounded(minimum: 0),
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self.sampler = sampler
        self.processor = processor
        self.toolCallInvocationRange = toolCallInvocationRange
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
        self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
      }
    }

    private struct State {
      let grammarEngine: NeedleXGrammarEngine
      let model: NeedleMLXModel
      let matcherPool: MatcherPool
    }

    private let state: Lock<State>
    private let _tokenizer: any Tokenizers.Tokenizer

    public var tokenizer: any Tokenizers.Tokenizer {
      self._tokenizer
    }

    public var model: NeedleMLXModel {
      self.state.withLock(\.model)
    }

    private let clock = ContinuousClock()

    public convenience init(
      from url: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      grammarEngine: (any Tokenizers.Tokenizer) -> NeedleXGrammarEngine? = {
        NeedleXGrammarEngine(tokenizer: $0)
      }
    ) throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: url.appending(path: "tokenizer.model"))
      let grammarEngine = grammarEngine(tokenizer)
      guard let grammarEngine else { throw NeedleMLXEngineError.failedToLoadGrammarEngine }

      var configuration = try JSONDecoder()
        .decode(
          NeedleModelConfiguration.self,
          from: Data(contentsOf: url.appending(path: "config.json"))
        )
      editConfiguration(&configuration)
      let model = NeedleMLXModel(configuration: configuration)
      try model.loadWeights(from: url.appending(path: "model.safetensors"))

      self.init(tokenizer: tokenizer, model: model, grammarEngine: grammarEngine)
    }

    public init(
      tokenizer: any Tokenizers.Tokenizer,
      model: sending NeedleMLXModel,
      grammarEngine: sending NeedleXGrammarEngine
    ) {
      self.state = Lock(
        State(grammarEngine: grammarEngine, model: model, matcherPool: MatcherPool())
      )
      self._tokenizer = tokenizer
    }

    public func tokenize(prompt: NeedlePrompt) async throws -> [NeedleToken] {
      prompt.tokenized(using: self._tokenizer)
    }

    public func clearCaches() {
      self.state.withLock { state in
        state.matcherPool.clear()
        state.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void
    ) throws -> GenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        // NB: This is safe, Swift must infer the worst when it can't infer isolation. We only
        // use the params within `generate` and not as a part of the return value.
        nonisolated(unsafe) let parameters = parameters
        return try self.state.withLock { state in
          try self.generate(
            prompt: prompt,
            parameters: parameters,
            onToken: onToken,
            state: &state,
            isStopped: isStopped
          )
        }
      }
      return GenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void,
      state: inout sending State,
      isStopped: ManagedAtomic<Bool>
    ) throws -> NeedleEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = try state.matcherPool.matcher(
        tools: prompt.tools,
        range: parameters.toolCallInvocationRange,
        compilingWith: state.grammarEngine
      )
      matcher.reset()
      state.model.reset()
      defer { state.model.reset() }

      let generationStartSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let generateStart = self.clock.now

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

      var detokenizer = StreamingDetokenizer(tokenizer: self._tokenizer)
      var generatedTokens = [NeedleToken]()
      var confidence = NeedleMLXConfidenceState()
      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self._tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        let processedLogits = processor?.process(logits: output.logits) ?? output.logits
        let logits = applyBitmaskMLX(
          logits: processedLogits[0..., -1, 0...],
          mask: matcher.bitmask()
        )
        confidence.add(logits: logits)

        let token = parameters.sampler.sample(logits: logits)
        let tokenId = token.item(NeedleToken.ID.self)

        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)
        let tokenString = detokenizer.decode(tokenId: tokenId)
        let needleToken = NeedleToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(needleToken)
        guard matcher.accept(tokenId: needleToken.id) else {
          throw NeedleMLXEngineError.grammarRejectedToken(token: needleToken)
        }
        onToken(needleToken)
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
      var metadata = NeedleMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = postPrefillSnapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = postDecodeSnapshot
      metadata.mlxEngineGenerationConfidence = confidence.mean
      metadata.mlxEnginePerTokenConfidences = confidence.perTokenConfidences
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
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
      model: NeedleMLXModel,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> (LMOutput?, NeedlePrefillMetrics, Memory.Snapshot) {
      let input = try LMInput.needle(prompt: prompt, using: self._tokenizer)
      guard input.text.tokens.size <= model.configuration.encoderMaxLength else {
        throw NeedleMLXEngineError.contextLengthExceeded(
          tokens: input.text.tokens.size,
          maximum: model.configuration.encoderMaxLength
        )
      }

      let prefillStart = self.clock.now
      processor?.prompt(input.text.tokens)
      let output = try model.prepare(input.text.tokens, cache: cache, windowSize: nil)
      let metrics = NeedlePrefillMetrics(
        tokens: input.text.tokens.size,
        duration: prefillStart.duration(to: self.clock.now)
      )
      let snapshot = Memory.synchronizedSnapshot(synchronize: synchronize)
      return (output, metrics, snapshot)
    }
  }

  // MARK: - MatcherPool

  extension NeedleMLXEngine {
    private final class MatcherPool {
      private struct Key: Hashable, Sendable {
        let tools: [NeedleToolDefinition]
        let range: NeedleXGrammarEngine.ToolCallInvocationRange
      }

      private let maxCount: Int
      private var entries = [Key: NeedleXGrammarEngine.Matcher]()
      private var order = [Key]()

      init(maxCount: Int = 8) {
        self.maxCount = maxCount
      }

      func matcher(
        tools: some Sequence<NeedleToolDefinition>,
        range: NeedleXGrammarEngine.ToolCallInvocationRange,
        compilingWith engine: NeedleXGrammarEngine
      ) throws -> NeedleXGrammarEngine.Matcher {
        let key = Key(tools: tools.map { $0.normalized() }, range: range)
        if let cached = self.entries[key] {
          self.touch(key)
          return cached
        }
        let matcher = try engine.compile(tools: key.tools, range: key.range)
        self.insert(key, matcher)
        return matcher
      }

      func clear() {
        self.entries.removeAll()
        self.order.removeAll()
      }

      private func touch(_ key: Key) {
        self.order.removeAll { $0 == key }
        self.order.append(key)
      }

      private func insert(_ key: Key, _ matcher: NeedleXGrammarEngine.Matcher) {
        if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
          self.entries.removeValue(forKey: leastRecentlyUsed)
          self.order.removeFirst()
        }
        self.entries[key] = matcher
        self.order.append(key)
      }
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

    public static let failedToLoadGrammarEngine = Self(message: "Could not load grammar engine.")

    public static func grammarRejectedToken(token: NeedleToken) -> Self {
      Self(
        message:
          "Token (ID=\(token.id), VALUE=\(token.stringValue)) was rejected by the grammar matcher."
      )
    }

    public static func contextLengthExceeded(tokens: Int, maximum: Int) -> Self {
      Self(message: "Prompt token count (\(tokens)) exceeds the model context length (\(maximum)).")
    }
  }
#endif
