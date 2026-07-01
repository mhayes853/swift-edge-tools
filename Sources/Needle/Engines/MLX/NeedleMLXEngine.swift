#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Tokenizers

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine, Sendable {
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
      var isStopped = false
      let grammarEngine: NeedleXGrammarEngine
      let model: NeedleMLXModel
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
      self.state = Lock(State(grammarEngine: grammarEngine, model: model))
      self._tokenizer = tokenizer
    }

    public func tokenize(prompt: NeedlePrompt) -> [NeedleToken] {
      prompt.tokenized(using: self._tokenizer)
    }

    public func stop() {
      self.state.withLock { $0.isStopped = true }
    }

    public func reset() {
      self.state.withLock {
        $0.model.reset()
        $0.isStopped = false
      }
    }

    public func clearCaches() {
      self.state.withLock { $0.grammarEngine.clearCache() }
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void
    ) async throws -> NeedleEngineGeneration {
      try Task.checkCancellation()
      guard !self.state.withLock({ $0.isStopped }) else { return .empty }
      let matcher = try self.state.withLock { state in
        state.isStopped = false
        return try state.grammarEngine.compile(
          tools: prompt.tools,
          range: parameters.toolCallInvocationRange
        )
      }

      let generationStartSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let generateStart = self.clock.now

      let model = self.state.withLock { $0.model }
      var processor = parameters.processor
      var cache = model.newCache(parameters: nil)
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

      var detokenizer = StreamingDetokenizer(tokenizer: self._tokenizer)
      var generatedTokens = [NeedleToken]()
      var confidence = NeedleMLXConfidenceState()
      while !matcher.isTerminated
        && !self.state.withLock({ $0.isStopped })
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
        output = model(inputText[text: .newAxis], cache: cache, state: output.state)
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
        wasStopped: self.state.withLock { $0.isStopped },
        tokens: generatedTokens,
        metadata: metadata
      )
    }

    private func prefill(
      prompt: NeedlePrompt,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> (LMOutput?, NeedlePrefillMetrics, Memory.Snapshot) {
      let model = self.state.withLock(\.model)
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
