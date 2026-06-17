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

    private let clock = ContinuousClock()
    private let sampler = ArgMaxSampler()

    public convenience init(
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

      let configuration = try JSONDecoder()
        .decode(
          NeedleModelConfiguration.self,
          from: Data(contentsOf: url.appending(path: "config.json"))
        )
      let weights = try MLX.loadArrays(url: url.appending(path: "model.safetensors"))
        .mapValues { $0.asType(configuration.mlxDType) }
      let model = NeedleMLXModel(configuration: configuration)
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
    }

    public func generate(
      prompt: NeedlePrompt,
      matcher: GrammarEngine.Matcher,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      let memoryUsage = MLXMemoryUsage()
      let generateStart = self.clock.now
      let (cache, prefillOutput, prefillMetrics) = try self.prefill(prompt: prompt)
      guard var output = prefillOutput else {
        preconditionFailure("Model received empty input.")
      }
      matcher.reset()
      var logits = output.logits
      var state = output.state
      var _durationToFirstToken: Duration?

      var tokenIds = [NeedleToken.ID]()
      while !matcher.isTerminated {
        let (token, needleToken) = self.sampleToken(logits: logits, using: matcher)
        _durationToFirstToken = _durationToFirstToken ?? generateStart.duration(to: self.clock.now)
        tokenIds.append(needleToken.id)
        guard matcher.accept(tokenId: needleToken.id) else {
          throw NeedleMLXEngineError.grammarRejectedToken(token: needleToken)
        }
        onToken(needleToken)

        let inputText = LMInput.Text(tokens: token)
        output = self.model(
          inputText[text: .newAxis],
          cache: cache,
          state: state
        )
        state = output.state
        logits = output.logits
      }

      let durationToFirstToken = _durationToFirstToken ?? .zero
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: tokenIds.count,
          duration: generateStart.duration(to: self.clock.now) - durationToFirstToken,
          durationToFirstToken: durationToFirstToken,
          ramUsageBytes: memoryUsage.stop()
        ),
        response: tokenizer.decode(tokenIds: tokenIds)
      )
    }

    public func reset() {
    }

    private func sampleToken(
      logits: MLXArray,
      using matcher: NeedleXGrammarEngine.Matcher
    ) -> (MLXArray, NeedleToken) {
      let logits = applyBitmaskMLX(logits: logits[0..., -1, 0...], mask: matcher.bitmask())
      let token = self.sampler.sample(logits: logits)
      let tokenId = token.item(NeedleToken.ID.self)
      let tokenString = self.tokenizer.decode(tokenIds: CollectionOfOne(tokenId))
      return (token, NeedleToken(id: tokenId, stringValue: tokenString))
    }

    private func prefill(
      prompt: NeedlePrompt
    ) throws -> ([any KVCache], LMOutput?, NeedlePrefillMetrics) {
      let input = try LMInput.needle(prompt: prompt, using: self.tokenizer)
      let cache = self.model.newCache(parameters: nil)
      let prefillStart = self.clock.now
      let (output, memoryUsage) = try withMemoryUsage {
        try self.model.prepare(input.text.tokens, cache: cache, windowSize: nil)
      }
      let metrics = NeedlePrefillMetrics(
        tokens: input.text.tokens.size,
        duration: prefillStart.duration(to: self.clock.now),
        ramUsageBytes: memoryUsage
      )
      return (cache, output, metrics)
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
