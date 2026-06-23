#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Atomics

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine {
    public struct GenerateParamaters: NeedleEngineGenerateParameters {
      public static var `default`: Self {
        Self()
      }

      public var sampler: LogitSampler
      public var processor: LogitProcessor?
      public var toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int

      public init(
        sampler: any LogitSampler = ArgMaxSampler(),
        processor: (any LogitProcessor)? = nil,
        toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange =
          .unbounded(minimum: 0),
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64
      ) {
        self.sampler = sampler
        self.processor = processor
        self.toolCallInvocationRange = toolCallInvocationRange
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
      }
    }

    public let grammarEngine: NeedleXGrammarEngine
    public let tokenizer: NeedleSPTokenizingModel
    public let model: NeedleMLXModel
    private var kvCache: [any KVCache]
    private let matcherPool: MatcherPool

    public var stopper: NeedleEngineStopper {
      let isStopped = self.isStopped
      return NeedleEngineStopper { isStopped.store(true, ordering: .relaxed) }
    }

    private let isStopped = ManagedAtomic<Bool>(false)
    private let clock = ContinuousClock()

    public convenience init(
      from url: URL,
      grammarEngine: (NeedleSPTokenizingModel) -> NeedleXGrammarEngine? = {
        NeedleXGrammarEngine(tokenizer: $0)
      }
    ) throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: url.appending(path: "tokenizer.model"))
      let grammarEngine = grammarEngine(tokenizer)
      guard let grammarEngine else { throw NeedleMLXEngineError.failedToLoadGrammarEngine }

      let configuration = try JSONDecoder()
        .decode(
          NeedleModelConfiguration.self,
          from: Data(contentsOf: url.appending(path: "config.json"))
        )
      var weights = try MLX.loadArrays(url: url.appending(path: "model.safetensors"))
      let model = NeedleMLXModel(configuration: configuration)
      weights = model.sanitize(weights: weights)
      try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)

      self.init(tokenizer: tokenizer, model: model, grammarEngine: grammarEngine)
    }

    public init(
      tokenizer: NeedleSPTokenizingModel,
      model: NeedleMLXModel,
      grammarEngine: NeedleXGrammarEngine
    ) {
      self.tokenizer = tokenizer
      self.model = model
      self.grammarEngine = grammarEngine
      self.kvCache = model.newCache(parameters: nil)
      self.matcherPool = MatcherPool()
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParamaters,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      self.isStopped.store(false, ordering: .relaxed)
      try Task.checkCancellation()
      let generationStartSnapshot = Memory.synchronizedSnapshot()
      let generateStart = self.clock.now

      let matcher = try self.matcherPool.matcher(
        tools: prompt.tools,
        range: parameters.toolCallInvocationRange,
        compilingWith: self.grammarEngine
      )

      var processor = parameters.processor
      let (prefillOutput, prefillMetrics, postPrefillSnapshot) =
        try self.prefill(prompt: prompt, processor: &processor)
      try Task.checkCancellation()
      guard var output = prefillOutput else { preconditionFailure("Model received empty input.") }
      matcher.reset()
      var _durationToFirstToken: Duration?

      var detokenizer = StreamingDetokenizer(tokenizer: self.tokenizer)
      var generatedTokens = [NeedleToken]()
      while !matcher.isTerminated
        && !self.isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
      {
        try Task.checkCancellation()
        let logits = processor?.process(logits: output.logits) ?? output.logits
        let (token, tokenId) = self.sampleToken(
          logits: logits,
          and: parameters.sampler,
          matcher: matcher
        )
        _durationToFirstToken = _durationToFirstToken ?? generateStart.duration(to: self.clock.now)
        let tokenString = detokenizer.decode(tokenId: tokenId)
        let needleToken = NeedleToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(needleToken)
        guard matcher.accept(tokenId: needleToken.id) else {
          throw NeedleMLXEngineError.grammarRejectedToken(token: needleToken)
        }
        onToken(needleToken)
        processor?.didSample(token: token)

        maybeQuantizeKVCache(
          cache: &self.kvCache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize
        )

        let inputText = LMInput.Text(tokens: token)
        output = self.model(inputText[text: .newAxis], cache: self.kvCache, state: output.state)
      }

      let durationToFirstToken = _durationToFirstToken ?? .zero
      let postDecodeSnapshot = Memory.synchronizedSnapshot()
      var metadata = NeedleMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = postPrefillSnapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = postDecodeSnapshot
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: generatedTokens.count,
          duration: generateStart.duration(to: self.clock.now) - durationToFirstToken,
          durationToFirstToken: durationToFirstToken
        ),
        wasStopped: self.isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        metadata: metadata
      )
    }

    public func reset() {
      self.kvCache = self.model.newCache(parameters: nil)
      self.model.reset()
      self.isStopped.store(false, ordering: .relaxed)
    }

    public func clearCaches() {
      self.kvCache = self.model.newCache(parameters: nil)
      self.matcherPool.clear()
      self.grammarEngine.clearCache()
    }

    private func sampleToken(
      logits: MLXArray,
      and sampler: any LogitSampler,
      matcher: NeedleXGrammarEngine.Matcher
    ) -> (MLXArray, NeedleToken.ID) {
      let logits = applyBitmaskMLX(logits: logits[0..., -1, 0...], mask: matcher.bitmask())
      let token = sampler.sample(logits: logits)
      let tokenId = token.item(NeedleToken.ID.self)
      return (token, tokenId)
    }

    private func prefill(
      prompt: NeedlePrompt,
      processor: inout (any LogitProcessor)?
    ) throws -> (LMOutput?, NeedlePrefillMetrics, Memory.Snapshot) {
      let input = try LMInput.needle(prompt: prompt, using: self.tokenizer)

      let prefillStart = self.clock.now
      processor?.prompt(input.text.tokens)
      let output = try self.model.prepare(input.text.tokens, cache: self.kvCache, windowSize: nil)
      let metrics = NeedlePrefillMetrics(
        tokens: input.text.tokens.size,
        duration: prefillStart.duration(to: self.clock.now)
      )
      let snapshot = Memory.synchronizedSnapshot()
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
        if self.entries.count >= self.maxCount, let lru = self.order.first {
          self.entries.removeValue(forKey: lru)
          self.order.removeFirst()
        }
        self.entries[key] = matcher
        self.order.append(key)
      }
    }
  }

  // MARK: - StreamingDetokenizer

  private struct StreamingDetokenizer {
    private static let replacementCharacter = "\u{fffd}"

    private let tokenizer: NeedleSPTokenizingModel
    private(set) var tokenIds = [NeedleToken.ID]()
    private(set) var streamedResponse = ""

    init(tokenizer: NeedleSPTokenizingModel) {
      self.tokenizer = tokenizer
    }

    mutating func decode(tokenId: NeedleToken.ID) -> String {
      self.tokenIds.append(tokenId)
      let decodedResponse = self.tokenizer.decode(tokenIds: self.tokenIds)
      guard decodedResponse.hasPrefix(self.streamedResponse) else {
        self.streamedResponse = decodedResponse
        return decodedResponse
      }

      let startIndex = decodedResponse.index(
        decodedResponse.startIndex,
        offsetBy: self.streamedResponse.count
      )
      let tokenString = String(decodedResponse[startIndex...])
      guard !tokenString.hasSuffix(Self.replacementCharacter) else { return "" }
      self.streamedResponse = decodedResponse
      return tokenString
    }
  }

  // MARK: - Synchronized Memory Snapshot

  extension Memory {
    fileprivate static func synchronizedSnapshot() -> Snapshot {
      Stream.defaultStream(.defaultDevice()).synchronize()
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
  }
#endif
