#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation

  // MARK: - NeedleMLXEngine

  public final class NeedleMLXEngine: NeedleEngine {
    public typealias GrammarEngine = NeedleXGrammarEngine

    public let grammarEngine: NeedleXGrammarEngine

    private let tokenizer: NeedleSPTokenizingModel
    private let model: NeedleMLXModel
    private let maskProcessor = BitaskLogitsProcessor()
    private var tokenIterator: TokenIterator?
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

    public func prefill(prompt: NeedlePrompt) throws -> NeedlePrefillMetrics {
      let (tokenIterator, metrics) = try self.loadTokenIterator(prompt: prompt)
      self.tokenIterator = tokenIterator
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

      var tokens = [NeedleToken]()
      for tokenId in tokenIterator {
        let token = NeedleToken(
          id: tokenId,
          stringValue: self.tokenizer.decode(tokenIds: CollectionOfOne(tokenId))
        )
        durationToFirstToken = durationToFirstToken ?? self.clock.now.duration(to: decodeStart)
        tokens.append(token)
        onToken(token)
      }

      let decodeDuration = self.clock.now.duration(to: decodeStart)
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: tokens.count,
          duration: decodeDuration,
          durationToFirstToken: durationToFirstToken ?? .zero,
          ramUsageBytes: 0
        ),
        response: tokens.map(\.stringValue).joined()
      )
    }

    public func reset() {
      self.tokenIterator = nil
      self.maskProcessor.use(matcher: nil)
    }

    private func loadGenerationTokenIterator(
      prompt: NeedlePrompt
    ) throws -> (TokenIterator, NeedlePrefillMetrics) {
      if let tokenIterator {
        let metrics = NeedlePrefillMetrics(tokens: 0, duration: .zero, ramUsageBytes: 0)
        return (tokenIterator, metrics)
      } else {
        return try self.loadTokenIterator(prompt: prompt)
      }
    }

    private func loadTokenIterator(
      prompt: NeedlePrompt
    ) throws -> (TokenIterator, NeedlePrefillMetrics) {
      let input = try LMInput.needle(prompt: prompt, using: self.tokenizer)
      var tokenIterator: TokenIterator!
      let prefillDuration = try self.clock.measure {
        tokenIterator = try TokenIterator(
          input: input,
          model: self.model,
          cache: nil,
          processor: self.maskProcessor,
          sampler: ArgMaxSampler()
        )
      }
      return (
        tokenIterator,
        NeedlePrefillMetrics(
          tokens: input.text.tokens.size,
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

  // MARK: - BitaskLogitsProcessor

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
