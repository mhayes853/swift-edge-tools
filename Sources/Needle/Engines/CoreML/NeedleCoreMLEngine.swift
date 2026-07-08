#if swift(>=6.4) && CoreML && Sentencepiece && canImport(CoreML)
  import Atomics
  @preconcurrency import CoreML
  import Foundation
  import Tokenizers

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public final class NeedleCoreMLEngine: NeedleEngine {
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
        get async throws { try await self.task.cancellableValue }
      }

      public func stop() {
        self.isStopped.store(true, ordering: .relaxed)
      }
    }

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

    private struct SendableModel: @unchecked Sendable {
      // NB: @unchecked Sendable is safe here because all MLModel access is serialized through
      // actor isolation, and decode state is reused behind the decoder actor.
      let value: MLModel
    }

    private actor EncoderModelActor {
      private let model: SendableModel

      init(model: SendableModel) {
        self.model = model
      }

      func prediction(from inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        try await self.model.value.prediction(from: inputs)
      }
    }

    private actor DecoderModelActor {
      private let model: SendableModel
      private let state: MLState

      init(model: SendableModel) {
        self.model = model
        self.state = model.value.makeState()
      }

      func reset() {
        self.state.zero(named: TensorName.keyCache)
        self.state.zero(named: TensorName.valueCache)
        self.state.zero(named: TensorName.cacheOffset)
      }

      func prediction(from inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        try await self.model.value.prediction(from: inputs, using: self.state)
      }
    }

    private actor GenerationGate {
      private var isGenerating = false
      private var waiters = [CheckedContinuation<Void, Never>]()

      func withExclusiveAccess<T: Sendable>(
        _ operation: @Sendable () async throws -> T
      ) async rethrows -> T {
        await self.acquire()
        defer { self.release() }
        return try await operation()
      }

      private func acquire() async {
        guard self.isGenerating else {
          self.isGenerating = true
          return
        }
        await withCheckedContinuation { continuation in
          self.waiters.append(continuation)
        }
      }

      private func release() {
        guard let waiter = self.waiters.first else {
          self.isGenerating = false
          return
        }
        self.waiters.removeFirst()
        waiter.resume()
      }
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderModel: EncoderModelActor
    private let decoderModel: DecoderModelActor
    private let tokenizer: any Tokenizer
    private let clock = ContinuousClock()
    private let generationGate = GenerationGate()

    public convenience init(
      modelDirectoryURL: URL,
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

      async let encoderModel = Self.loadModel(named: ModelName.encoder, from: modelDirectoryURL)
      async let decoderModel = Self.loadModel(named: ModelName.decoder, from: modelDirectoryURL)
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: tokenizer,
        configuration: configuration,
        grammarEngine: grammarEngine
      )
    }

    public init(
      encoderModel: MLModel,
      decoderModel: MLModel,
      tokenizer: any Tokenizer,
      configuration: NeedleModelConfiguration,
      grammarEngine: sending NeedleXGrammarEngine
    ) {
      self.state = Lock(
        State(grammarEngine: grammarEngine, matcherPool: NeedleGrammarMatcherPool())
      )
      self.configuration = configuration
      self.encoderModel = EncoderModelActor(model: SendableModel(value: encoderModel))
      self.decoderModel = DecoderModelActor(model: SendableModel(value: decoderModel))
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
    ) throws -> GenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        try await self.generationGate.withExclusiveAccess {
          let matcher = try self.state.withLock { state in
            let matcher = try state.matcherPool.matcher(
              tools: prompt.tools,
              range: parameters.toolCallRange,
              compilingWith: state.grammarEngine
            )
            matcher.reset()
            return matcher
          }
          return try await self.generate(
            prompt: prompt,
            parameters: parameters,
            onToken: onToken,
            matcher: matcher,
            configuration: self.configuration,
            isStopped: isStopped
          )
        }
      }
      return GenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void,
      matcher: NeedleXGrammarEngine.Matcher,
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

      await self.decoderModel.reset()
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
          encoderOutputs: encoderOutputs
        )
        var logits = try await self.stepLogits(from: decoderLogits)
        let processedLogits = processor?.process(logits: &logits) ?? logits
        let maskedLogits = Self.applyBitmask(processedLogits, mask: matcher.bitmask())
        confidence.add(confidence: Self.tokenConfidence(maskedLogits))

        let tokenId = sampler.sample(logits: maskedLogits)
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
      encoderOutputs: EncoderOutputs
    ) async throws -> MLTensor {
      var outputs = try await self.decoderModel.prediction(
        from: [
          TensorName.inputIds: inputIds,
          TensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
          TensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
          TensorName.encoderProjectedV: encoderOutputs.encoderProjectedV
        ]
      )
      guard let logits = outputs.removeValue(forKey: TensorName.logits) else {
        throw NeedleCoreMLEngineError.missingModelOutputs
      }
      return logits
    }

    private func stepLogits(from logits: MLTensor) async throws -> [Float] {
      let scalars = await logits.cast(to: Float.self).shapedArray(of: Float.self)
      let stepIndex = logits.shape[1] - 1
      return (0..<logits.shape[2]).map { scalars[scalarAt: [0, stepIndex, $0]] }
    }

    private static func decodeConfiguration(
      from directory: URL
    ) throws -> NeedleModelConfiguration {
      guard let config = try NeedleModelConfiguration.decode(in: directory) else {
        throw NeedleCoreMLEngineError.failedToLoadConfiguration
      }
      return config
    }

    private static func loadModel(named name: String, from directory: URL) async throws -> MLModel {
      let packageURL = directory.appending(path: "\(name).mlpackage")
      let configuration = MLModelConfiguration()
      configuration.computeUnits = .cpuAndGPU
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

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    public protocol Sampler {
      func sample(logits: [Float]) -> NeedleToken.ID
    }

    public struct ArgmaxSampler: Sampler {
      public init() {}

      public func sample(logits: [Float]) -> NeedleToken.ID {
        logits.indices.max { logits[$0] < logits[$1] } ?? 0
      }
    }
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    public protocol LogitsProcessor {
      mutating func prompt(_ prompt: MLTensor)

      func process(logits: inout [Float]) -> [Float]

      mutating func didSample(token: NeedleToken)
    }
  }

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

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    private static func applyBitmask(
      _ logits: [Float],
      mask: NeedleGrammarBitmask
    ) -> [Float] {
      let maskValues = mask.storage.withUnsafeBytes { bytes in
        bytes.bindMemory(to: UInt8.self)
          .flatMap { Needle.bitmaskTable[(Int($0) * 8)..<(Int($0) * 8 + 8)] }
      }
      return zip(logits, maskValues).map(+)
    }

    private static func tokenConfidence(_ logits: [Float]) -> Float {
      let topTwo = logits.sorted(by: >).prefix(2)
      let values = Array(topTwo)
      let top1 = values.first ?? -.infinity
      let top2 = values.dropFirst().first ?? -.infinity
      let margin = Swift.min(Swift.max(top1 - top2, -60.0), 60.0)
      return 1.0 / (1.0 + Foundation.exp(-margin))
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
