#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Atomics
  import Tokenizers

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine {
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

    public let grammarEngine: NeedleXGrammarEngine
    public let tokenizer: any Tokenizers.Tokenizer
    public let model: NeedleMLXModel
    private let matcherPool: MatcherPool

    public var stopper: NeedleEngineStopper {
      let isStopped = self.isStopped
      return NeedleEngineStopper { isStopped.store(true, ordering: .relaxed) }
    }

    private let isStopped = ManagedAtomic<Bool>(false)
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
      model: NeedleMLXModel,
      grammarEngine: NeedleXGrammarEngine
    ) {
      self.tokenizer = tokenizer
      self.model = model
      self.grammarEngine = grammarEngine
      self.matcherPool = MatcherPool()
    }

    public func tokenize(prompt: NeedlePrompt) -> [NeedleToken] {
      prompt.tokenized(using: self.tokenizer)
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParameters,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      try Task.checkCancellation()
      guard !self.isStopped.load(ordering: .relaxed) else { return .empty }
      let generationStartSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let generateStart = self.clock.now

      let matcher = try self.matcherPool.matcher(
        tools: prompt.tools,
        range: parameters.toolCallInvocationRange,
        compilingWith: self.grammarEngine
      )

      var processor = parameters.processor
      var cache = self.model.newCache(parameters: nil)
      let (prefillOutput, prefillMetrics, postPrefillSnapshot) = try self.prefill(
        prompt: prompt,
        cache: cache,
        processor: &processor,
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      try Task.checkCancellation()
      guard var output = prefillOutput else { preconditionFailure("Model received empty input.") }
      matcher.reset()
      var _durationToFirstToken: Duration?

      var detokenizer = StreamingDetokenizer(tokenizer: self.tokenizer)
      var generatedTokens = [NeedleToken]()
      var confidence = NeedleMLXConfidenceState()
      while !matcher.isTerminated
        && !self.isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.eosTokenId
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
          cache: &cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize
        )

        let inputText = LMInput.Text(tokens: token)
        output = self.model(inputText[text: .newAxis], cache: cache, state: output.state)
      }

      let durationToFirstToken = _durationToFirstToken ?? .zero
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
          duration: generateStart.duration(to: self.clock.now) - durationToFirstToken,
          durationToFirstToken: durationToFirstToken
        ),
        wasStopped: self.isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        metadata: metadata
      )
    }

    public func reset() {
      self.model.reset()
      self.isStopped.store(false, ordering: .relaxed)
    }

    public func clearCaches() {
      self.matcherPool.clear()
      self.grammarEngine.clearCache()
    }

    private func prefill(
      prompt: NeedlePrompt,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> (LMOutput?, NeedlePrefillMetrics, Memory.Snapshot) {
      let input = try LMInput.needle(prompt: prompt, using: self.tokenizer)
      guard input.text.tokens.size <= self.model.configuration.encoderMaxLength else {
        throw NeedleMLXEngineError.contextLengthExceeded(
          tokens: input.text.tokens.size,
          maximum: self.model.configuration.encoderMaxLength
        )
      }

      let prefillStart = self.clock.now
      processor?.prompt(input.text.tokens)
      let output = try self.model.prepare(input.text.tokens, cache: cache, windowSize: nil)
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

    private let tokenizer: any Tokenizers.Tokenizer
    private(set) var tokenIds = [NeedleToken.ID]()
    private(set) var streamedResponse = ""

    init(tokenizer: any Tokenizers.Tokenizer) {
      self.tokenizer = tokenizer
    }

    mutating func decode(tokenId: NeedleToken.ID) -> String {
      self.tokenIds.append(tokenId)
      let decodedResponse = self.tokenizer.decode(tokens: self.tokenIds)
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
