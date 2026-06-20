#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation
  import Atomics

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine {
    public typealias GrammarEngine = NeedleXGrammarEngine

    public struct GenerateParamaters: NeedleEngineGenerateParameters {
      public static var `default`: Self {
        Self()
      }

      public var sampler: LogitSampler
      public var processor: LogitProcessor?
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int

      public init(
        sampler: any LogitSampler = ArgMaxSampler(),
        processor: (any LogitProcessor)? = nil,
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64
      ) {
        self.sampler = sampler
        self.processor = processor
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
      }
    }

    public let grammarEngine: NeedleXGrammarEngine
    public let tokenizer: NeedleSPTokenizingModel
    public let model: NeedleMLXModel
    private var cache: [any KVCache]

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
      self.cache = model.newCache(parameters: nil)
    }

    public func generate(
      prompt: NeedlePrompt,
      matcher: GrammarEngine.Matcher,
      parameters: GenerateParamaters,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      self.isStopped.store(false, ordering: .relaxed)
      try Task.checkCancellation()
      let memoryUsage = MLXMemoryUsage()
      let generateStart = self.clock.now

      var processor = parameters.processor
      let (prefillOutput, prefillMetrics) = try self.prefill(prompt: prompt, processor: &processor)
      try Task.checkCancellation()
      guard var output = prefillOutput else { preconditionFailure("Model received empty input.") }
      matcher.reset()
      var _durationToFirstToken: Duration?

      var detokenizer = StreamingDetokenizer(tokenizer: self.tokenizer)
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
        guard matcher.accept(tokenId: needleToken.id) else {
          throw NeedleMLXEngineError.grammarRejectedToken(token: needleToken)
        }
        onToken(needleToken)
        processor?.didSample(token: token)

        maybeQuantizeKVCache(
          cache: &self.cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize
        )

        let inputText = LMInput.Text(tokens: token)
        output = self.model(inputText[text: .newAxis], cache: self.cache, state: output.state)
      }

      let durationToFirstToken = _durationToFirstToken ?? .zero
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: detokenizer.tokenIds.count,
          duration: generateStart.duration(to: self.clock.now) - durationToFirstToken,
          durationToFirstToken: durationToFirstToken,
          ramUsageBytes: memoryUsage.stop()
        ),
        wasStopped: self.isStopped.load(ordering: .relaxed),
        response: self.tokenizer.decode(tokenIds: detokenizer.tokenIds),
        metadata: [:]
      )
    }

    public func reset() {
      self.cache = self.model.newCache(parameters: nil)
      self.isStopped.store(false, ordering: .relaxed)
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
    ) throws -> (LMOutput?, NeedlePrefillMetrics) {
      let input = try LMInput.needle(prompt: prompt, using: self.tokenizer)

      let prefillStart = self.clock.now
      processor?.prompt(input.text.tokens)
      let (output, memoryUsage) = try withMemoryUsage {
        try self.model.prepare(input.text.tokens, cache: self.cache, windowSize: nil)
      }
      let metrics = NeedlePrefillMetrics(
        tokens: input.text.tokens.size,
        duration: prefillStart.duration(to: self.clock.now),
        ramUsageBytes: memoryUsage
      )
      return (output, metrics)
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
