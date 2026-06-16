#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine {
    public typealias GrammarEngine = NeedleXGrammarEngine

    public let grammarEngine: NeedleXGrammarEngine
    public let tokenizer: NeedleSPTokenizingModel
    public let model: NeedleMLXModel

    private let maskProcessor = BitaskLogitsProcessor()
    private var prefilledInput: MLXArray?
    private var cache: [any KVCache]?
    private let clock = ContinuousClock()

    public init(
      from url: URL,
      grammarConfiguration: NeedleXGrammarEngine.Configuration =
        NeedleXGrammarEngine.Configuration(),
      makeGrammarEngine: (
        NeedleSPTokenizingModel,
        NeedleXGrammarEngine.Configuration
      ) -> NeedleXGrammarEngine? = { NeedleXGrammarEngine(tokenizer: $0, configuration: $1) }
    ) throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: url.appending(path: "tokenizer.model"))
      let grammarEngine = makeGrammarEngine(tokenizer, grammarConfiguration)
      guard let grammarEngine else { throw NeedleMLXEngineError.failedToLoadGrammarEngine }
      self.grammarEngine = grammarEngine

      let configuration = try JSONDecoder()
        .decode(
          NeedleModelConfiguration.self,
          from: Data(contentsOf: url.appending(path: "config.json"))
        )
      let weights = try MLX.loadArrays(url: url.appending(path: "model.safetensors"))
        .mapValues { $0.asType(configuration.mlxDType) }
      self.model = NeedleMLXModel(configuration: configuration)
      try self.model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)

      self.tokenizer = tokenizer
    }

    public func prefill(prompt: NeedlePrefillablePrompt) throws -> NeedlePrefillMetrics {
      self.cache = self.model.newCache(parameters: nil)
      let input = LMInput.needle(prompt: prompt, using: self.tokenizer)
      let (_, metrics) = try self.loadTokenIterator(input: input)
      self.prefilledInput = input.text.tokens
      return metrics
    }

    public func generate(
      prompt: NeedlePrompt,
      matcher: GrammarEngine.Matcher,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      self.maskProcessor.use(matcher: matcher)
      let (tokenIterator, prefillMetrics) = try self.loadGenerationTokenIterator(prompt: prompt)
      let decodeStart = self.clock.now
      var durationToFirstToken: Duration?

      var tokenIds = [NeedleToken.ID]()
      for tokenId in tokenIterator {
        let token = NeedleToken(
          id: tokenId,
          stringValue: self.tokenizer.decode(tokenIds: CollectionOfOne(tokenId))
        )
        durationToFirstToken = durationToFirstToken ?? decodeStart.duration(to: self.clock.now)
        tokenIds.append(token.id)
        onToken(token)

        if matcher.isTerminated {
          break
        }
      }

      let decodeDuration = decodeStart.duration(to: self.clock.now)
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: tokenIds.count,
          duration: decodeDuration,
          durationToFirstToken: durationToFirstToken ?? .zero,
          ramUsageBytes: 0
        ),
        response: tokenizer.decode(tokenIds: tokenIds)
      )
    }

    public func reset() {
      self.prefilledInput = nil
      self.cache = nil
      self.maskProcessor.use(matcher: nil)
    }

    private func loadGenerationTokenIterator(
      prompt: NeedlePrompt
    ) throws -> (TokenIterator, NeedlePrefillMetrics) {
      let input = try LMInput.needle(prompt: prompt, using: self.tokenizer)
      guard let prefilledInput else { return try self.loadTokenIterator(input: input) }

      let isPrefilledPrefix = all(prefilledInput .== input.text.tokens[0..<prefilledInput.dim(0)])
      if !isPrefilledPrefix.item(Bool.self) {
        self.cache = nil
      }
      return try self.loadTokenIterator(input: input)
    }

    private func loadTokenIterator(
      input: LMInput
    ) throws -> (TokenIterator, NeedlePrefillMetrics) {
      var tokenIterator: TokenIterator!
      let offset = self.cache?.first?.offset ?? 0
      let prefillDuration = try self.clock.measure {
        tokenIterator = try TokenIterator(
          input: input,
          model: self.model,
          cache: self.cache,
          processor: self.maskProcessor,
          sampler: ArgMaxSampler()
        )
      }
      return (
        tokenIterator,
        NeedlePrefillMetrics(
          tokens: input.text.tokens.size - offset,
          duration: prefillDuration,
          ramUsageBytes: 0
        )
      )
    }
  }

  // MARK: - NeedleMLXEngineError

  public struct NeedleMLXEngineError: Hashable, Error {
    public static let failedToLoadGrammarEngine = Self()
  }

  // MARK: - Helpers

  private final class BitaskLogitsProcessor: LogitProcessor {
    private var base: NeedleGrammarLogitsProcessor?

    func use(matcher: NeedleXGrammarEngine.Matcher?) {
      self.base = matcher.map { NeedleGrammarLogitsProcessor(matcher: $0) }
    }

    func prompt(_ prompt: MLXArray) {
      self.base?.prompt(prompt)
    }

    func process(logits: MLXArray) -> MLXArray {
      self.base?.process(logits: logits) ?? logits
    }

    func didSample(token: MLXArray) {
      self.base?.didSample(token: token)
    }
  }
#endif
