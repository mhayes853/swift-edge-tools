#if CoreML && Sentencepiece && canImport(CoreML)
  import Atomics
  @preconcurrency import CoreML
  import Foundation
  import Tokenizers

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public final class NeedleCoreMLEngine: NeedleEngine {
    public struct GenerateParameters: NeedleEngineGenerateParameters {
      public static var `default`: Self { Self() }

      private var _sampler: @Sendable () -> any Sampler
      private var _processor: @Sendable () -> (any LogitsProcessor)?

      public var sampler: any Sampler {
        self._sampler()
      }

      public var processor: (any LogitsProcessor)? {
        self._processor()
      }

      public var toolCallRange: NeedleGrammarToolCallRange
      public var maxTokens: Int?

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any Sampler = ArgmaxSampler(),
        processor: @autoclosure @escaping @Sendable () -> (any LogitsProcessor)? = nil,
        toolCallRange: NeedleGrammarToolCallRange = .unbounded(minimum: 0),
        maxTokens: Int? = 1024
      ) {
        self._sampler = sampler
        self._processor = processor
        self.toolCallRange = toolCallRange
        self.maxTokens = maxTokens
      }
    }

    private struct State {
      let grammarEngine: NeedleXGrammarEngine
      let matcherPool: NeedleGrammarMatcherPool
    }

    private struct EncoderOutputs {
      let crossAttentionMask: MLTensor
      let encoderProjectedK: MLTensor
      let encoderProjectedV: MLTensor
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderModel: EncoderModelActor
    private let decoderModel: DecoderModelActor
    private let tokenizer: any Tokenizer
    private let clock = ContinuousClock()

    public convenience init(
      modelDirectoryURL: URL,
      modelConfiguration: MLModelConfiguration,
      editModelConfiguration: (inout MLModelConfiguration) -> Void = { _ in },
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      grammarEngine: (any Tokenizer) -> NeedleXGrammarEngine? = {
        NeedleXGrammarEngine(tokenizer: $0)
      }
    ) async throws {
      let tokenizer = try NeedleSPTokenizer(
        modelURL: modelDirectoryURL.appending(path: "tokenizer.model")
      )
      guard let grammarEngine = grammarEngine(tokenizer) else {
        throw NeedleCoreMLEngineError.failedToLoadGrammarEngine
      }

      var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
      editConfiguration(&configuration)

      async let encoderModel = Self.loadModel(
        named: ModelName.encoder,
        from: modelDirectoryURL,
        configuration: modelConfiguration
      )
      async let decoderModel = Self.loadModel(
        named: ModelName.decoder,
        from: modelDirectoryURL,
        configuration: modelConfiguration
      )
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: tokenizer,
        configuration: configuration,
        grammarEngine: grammarEngine
      )
    }

    public init(
      encoderModel: sending MLModel,
      decoderModel: sending MLModel,
      tokenizer: any Tokenizer,
      configuration: NeedleModelConfiguration,
      grammarEngine: sending NeedleXGrammarEngine
    ) {
      self.state = Lock(
        State(grammarEngine: grammarEngine, matcherPool: NeedleGrammarMatcherPool())
      )
      self.configuration = configuration
      self.encoderModel = EncoderModelActor(model: encoderModel)
      self.decoderModel = DecoderModelActor(model: decoderModel)
      self.tokenizer = tokenizer
    }

    public func tokenize(prompt: NeedlePrompt) async throws -> [NeedleToken] {
      prompt.tokenized(using: self.tokenizer)
    }

    public func clearCaches() {
      self.state.withLock {
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void
    ) throws -> some NeedleEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            tools: prompt.tools,
            range: parameters.toolCallRange,
            compilingWith: state.grammarEngine
          )
          matcher.reset()
          return matcher
        }
        let decoderState = await self.decoderModel.beginDecodeWithState()
        do {
          let result = try await self.generate(
            prompt: prompt,
            parameters: parameters,
            onToken: onToken,
            matcher: matcher,
            decoderState: decoderState.mlState,
            configuration: self.configuration,
            isStopped: isStopped
          )
          await self.decoderModel.endDecodeWithState(decoderState)
          return result
        } catch {
          await self.decoderModel.endDecodeWithState(decoderState)
          throw error
        }
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void,
      matcher: NeedleXGrammarEngine.Matcher,
      decoderState: MLState,
      configuration: NeedleModelConfiguration,
      isStopped: ManagedAtomic<Bool>
    ) async throws -> NeedleEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let sampler = parameters.sampler
      var processor = parameters.processor
      let generateStart = self.clock.now
      let (encoderOutputs, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        configuration: configuration,
        processor: &processor
      )

      var nextDecoderTokenId = configuration.decoderStartTokenId
      var detokenizer = StreamingDetokenizer(tokenizer: self.tokenizer)
      var generatedTokens = [NeedleToken]()
      var confidence = NeedleConfidenceState()
      var durationToFirstToken: Duration?

      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        let decoderLogits = try await self.decode(
          inputIds: MLTensor(tokenIds: [nextDecoderTokenId]),
          encoderOutputs: encoderOutputs,
          decoderState: decoderState
        )
        let stepLogits = decoderLogits.squeezingShape(at: 1)
        let processedLogits = try await processor?.process(stepLogits) ?? stepLogits
        let maskedLogits = applyBitmaskCoreML(logits: processedLogits, mask: matcher.bitmask())
        await confidence.add(logits: maskedLogits)

        let tokenId = try await sampler.sample(maskedLogits)
        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)

        let tokenString = detokenizer.decode(tokenId: tokenId)
        let token = NeedleToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(token)
        nextDecoderTokenId = tokenId
        guard matcher.accept(tokenId: token.id) else {
          throw NeedleCoreMLEngineError.grammarRejectedToken(token: token)
        }
        onToken(token)
        processor?.didSample(token: token)
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      var metadata = NeedleMetadata()
      metadata.generationConfidence = confidence.mean
      metadata.perTokenConfidences = confidence.perTokenConfidences
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
      configuration: NeedleModelConfiguration,
      processor: inout (any LogitsProcessor)?
    ) async throws -> (EncoderOutputs, NeedlePrefillMetrics) {
      let promptTokens = self.tokenizer.encode(text: prompt.formatted())
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw NeedleCoreMLEngineError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: configuration.encoderMaxLength
        )
      }

      let promptTensor = MLTensor(tokenIds: promptTokens)
      let prefillStart = self.clock.now
      processor?.prompt(promptTensor)
      let encoderOutputs = try await self.runEncoder(inputIds: promptTensor)
      let metrics = NeedlePrefillMetrics(
        tokens: promptTokens.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      return (encoderOutputs, metrics)
    }

    private func runEncoder(inputIds: MLTensor) async throws -> EncoderOutputs {
      var outputs = try await self.encoderModel.prediction(from: [TensorName.inputIds: inputIds])
      guard
        let crossAttentionMask = outputs.removeValue(forKey: TensorName.crossAttentionMask),
        let encoderProjectedK = outputs.removeValue(forKey: TensorName.encoderProjectedK),
        let encoderProjectedV = outputs.removeValue(forKey: TensorName.encoderProjectedV)
      else {
        throw NeedleCoreMLEngineError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMask.cast(to: Float16.self),
        encoderProjectedK: encoderProjectedK.cast(to: Float16.self),
        encoderProjectedV: encoderProjectedV.cast(to: Float16.self)
      )
    }

    private func decode(
      inputIds: MLTensor,
      encoderOutputs: EncoderOutputs,
      decoderState: MLState
    ) async throws -> MLTensor {
      var outputs = try await self.decoderModel.prediction(
        from: [
          TensorName.inputIds: inputIds,
          TensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
          TensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
          TensorName.encoderProjectedV: encoderOutputs.encoderProjectedV
        ],
        using: decoderState
      )
      guard let logits = outputs.removeValue(forKey: TensorName.logits) else {
        throw NeedleCoreMLEngineError.missingModelOutputs
      }
      return logits
    }

    private static func decodeConfiguration(
      from directory: URL
    ) throws -> NeedleModelConfiguration {
      guard let config = try NeedleModelConfiguration.decode(in: directory) else {
        throw NeedleCoreMLEngineError.failedToLoadConfiguration
      }
      return config
    }

    private static func loadModel(
      named name: String,
      from directory: URL,
      configuration: MLModelConfiguration
    ) async throws -> MLModel {
      let packageURL = directory.appending(path: "\(name).mlpackage")
      #if os(watchOS)
        return try await MLModel.load(contentsOf: packageURL, configuration: configuration)
      #else
        let compiledURL = directory.appending(path: "\(name).mlmodelc")
        if !FileManager.default.fileExists(atPath: compiledURL.path()) {
          let temporaryCompiledURL = try await MLModel.compileModel(at: packageURL)
          if FileManager.default.fileExists(atPath: compiledURL.path()) {
            try FileManager.default.removeItem(at: compiledURL)
          }
          try FileManager.default.copyItem(at: temporaryCompiledURL, to: compiledURL)
        }
        return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
      #endif
    }
  }

  // MARK: - Encoder + Decoder Actors

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    private actor EncoderModelActor {
      private let model: MLModel

      init(model: sending MLModel) {
        self.model = model
      }

      func prediction(from inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        try await self.model.prediction(from: inputs)
      }
    }

    private actor DecoderModelActor {
      private nonisolated let model: MLModel
      private var bufferedState: State?

      struct State {
        let mlState: MLState
        let isPersistent: Bool

        func reset() {
          self.mlState.zero(named: TensorName.keyCache)
          self.mlState.zero(named: TensorName.valueCache)
          self.mlState.zero(named: TensorName.cacheOffset)
        }
      }

      init(model: sending MLModel) {
        self.model = model
        self.bufferedState = State(mlState: model.makeState(), isPersistent: true)
      }

      func beginDecodeWithState() -> sending State {
        if let persistent = self.bufferedState {
          self.bufferedState = nil
          persistent.reset()
          return persistent
        } else {
          return State(mlState: model.makeState(), isPersistent: false)
        }
      }

      func endDecodeWithState(_ state: sending State) {
        guard state.isPersistent else { return }
        self.bufferedState = state
      }

      func prediction(
        from inputs: [String: MLTensor],
        using state: MLState
      ) async throws -> [String: MLTensor] {
        try await self.model.prediction(from: inputs, using: state)
      }
    }
  }

  // MARK: - Sampler

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    public protocol Sampler {
      func sample(_ logits: MLTensor) async throws -> NeedleToken.ID
    }

    public struct ArgmaxSampler: Sampler {
      public init() {}

      public func sample(_ logits: MLTensor) async throws -> NeedleToken.ID {
        let indices = await logits.argmax(alongAxis: 1).shapedArray(of: Int32.self).scalars
        return NeedleToken.ID(indices.first ?? 0)
      }
    }
  }

  // MARK: - LogitsProcessor

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    public protocol LogitsProcessor {
      mutating func prompt(_ prompt: MLTensor)

      func process(_ logits: MLTensor) async throws -> MLTensor

      mutating func didSample(token: NeedleToken)
    }
  }

  // MARK: - Error

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public struct NeedleCoreMLEngineError: Hashable, Error {
    public let message: String

    public static let failedToLoadConfiguration = Self(
      message: "Could not load model configuration."
    )

    public static let failedToLoadGrammarEngine = Self(
      message: "Could not load grammar engine."
    )

    public static func grammarRejectedToken(token: NeedleToken) -> Self {
      Self(
        message:
          "Token (ID=\(token.id), VALUE=\(token.stringValue)) was rejected by the grammar matcher."
      )
    }

    public static func contextLengthExceeded(tokens: Int, maximum: Int) -> Self {
      Self(message: "Prompt token count (\(tokens)) exceeds the model context length (\(maximum)).")
    }

    public static let missingModelOutputs = Self(
      message: "CoreML model did not return expected outputs."
    )
  }

  // MARK: - Helpers

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension MLTensor {
    fileprivate init(tokenIds: some Sequence<NeedleToken.ID>) {
      let ids = tokenIds.map(Int32.init)
      self.init(shape: [1, ids.count], scalars: ids, scalarType: Int32.self)
    }
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension MLState {
    fileprivate func zero(named name: String) {
      self.withMultiArray(for: name) { array in
        array.withUnsafeMutableBytes { bytes, _ in
          guard let baseAddress = bytes.baseAddress else { return }
          memset(baseAddress, 0, bytes.count)
        }
      }
    }
  }

  private enum ModelName {
    static let encoder = "encoder"
    static let decoder = "decoder"
  }

  private enum TensorName {
    static let inputIds = "input_ids"
    static let crossAttentionMask = "cross_attention_mask"
    static let encoderProjectedK = "encoder_projected_k"
    static let encoderProjectedV = "encoder_projected_v"
    static let keyCache = "keyCache"
    static let valueCache = "valueCache"
    static let cacheOffset = "cacheOffset"
    static let logits = "logits"
  }
#endif
