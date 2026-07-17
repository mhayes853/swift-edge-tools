#if CoreML && canImport(CoreML)
  import Atomics
  @preconcurrency import CoreML
  import Foundation

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public final class NeedleCoreMLEngine: EdgeToolsEngine {
    public typealias Prompt = NeedlePrompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self { Self() }

      private var _sampler: @Sendable () -> any EdgeToolsSampler<MLTensor>
      private var _processor: @Sendable () -> (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?

      public var sampler: any EdgeToolsSampler<MLTensor> {
        self._sampler()
      }

      public var processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)? {
        self._processor()
      }

      public var toolCallRange: GrammarToolCallRange
      public var maxTokens: Int?

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any EdgeToolsSampler<MLTensor> =
          CoreMLArgmaxSampler(),
        processor:
          @autoclosure @escaping @Sendable () -> (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)? =
          nil,
        toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
        maxTokens: Int? = 1024
      ) {
        self._sampler = sampler
        self._processor = processor
        self.toolCallRange = toolCallRange
        self.maxTokens = maxTokens
      }
    }

    private struct State: ~Copyable {
      let grammarEngine: XGrammarCompiler
      let matcherPool: XGrammarToolCallMatcherPool
    }

    private struct EncoderOutputs {
      let crossAttentionMask: MLTensor
      let encoderProjectedK: MLTensor
      let encoderProjectedV: MLTensor
    }

    private struct DecoderCache: Sendable {
      var key: MLTensor
      var value: MLTensor

      init(key: MLTensor, value: MLTensor) {
        self.key = key
        self.value = value
      }

      init(configuration: NeedleModelConfiguration) {
        let shape = [
          configuration.decoderLayers,
          configuration.encoderMaxLength,
          configuration.attentionHeads,
          configuration.attentionHeadDimensions
        ]
        self.key = MLTensor(repeating: Float16.zero, shape: shape, scalarType: Float16.self)
        self.value = MLTensor(repeating: Float16.zero, shape: shape, scalarType: Float16.self)
      }
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderModel: EncoderModelActor
    private let decoderModel: DecoderModelActor
    private let tokenizer: any EdgeToolsTokenizer
    private let clock = ContinuousClock()

    public convenience init(
      modelDirectoryURL: URL,
      modelConfiguration: MLModelConfiguration,
      editModelConfiguration: (inout MLModelConfiguration) -> Void = { _ in },
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(
        from: modelDirectoryURL,
        isNeedleModel: true
      )
      let grammarEngine = try? XGrammarCompiler(
        tokenizerInfo: try XGrammarTokenizerInfo.needle(erasedTokenizer: tokenizer)
      )
      guard let grammarEngine else {
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
        tokenizer: consume tokenizer,
        configuration: configuration,
        grammarEngine: grammarEngine
      )
    }

    public convenience init<Tokenizer: EdgeToolsTokenizer>(
      encoderModel: sending MLModel,
      decoderModel: sending MLModel,
      tokenizer: consuming sending Tokenizer,
      configuration: NeedleModelConfiguration,
      grammarEngine: consuming sending XGrammarCompiler
    ) {
      self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: consume tokenizer,
        configuration: configuration,
        grammarEngine: consume grammarEngine
      )
    }

    private init(
      encoderModel: sending MLModel,
      decoderModel: sending MLModel,
      tokenizer: consuming sending any EdgeToolsTokenizer,
      configuration: NeedleModelConfiguration,
      grammarEngine: consuming sending XGrammarCompiler
    ) {
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          matcherPool: XGrammarToolCallMatcherPool.needle()
        )
      )
      self.configuration = configuration
      self.encoderModel = EncoderModelActor(model: encoderModel)
      self.decoderModel = DecoderModelActor(model: decoderModel)
      self.tokenizer = consume tokenizer
    }

    public func tokenize(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      try prompt.tokenized(tools: tools, using: self.tokenizer)
    }

    public func clearCaches() {
      self.state.withLock {
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition] = [],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            tools: tools,
            range: parameters.toolCallRange,
            compilingWith: state.grammarEngine
          )
          matcher.reset()
          return matcher
        }
        return try await self.generate(
          prompt: prompt,
          tools: tools,
          parameters: parameters,
          channel: channel,
          matcher: matcher,
          configuration: self.configuration,
          isStopped: isStopped
        )
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel,
      matcher: consuming XGrammarMatcher,
      configuration: NeedleModelConfiguration,
      isStopped: ManagedAtomic<Bool>
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = consume matcher
      let sampler = parameters.sampler
      var processor = parameters.processor
      let generateStart = self.clock.now
      let (encoderOutputs, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        tools: tools,
        configuration: configuration,
        processor: &processor
      )

      var cache = DecoderCache(configuration: configuration)
      var nextDecoderTokenId = configuration.decoderStartTokenId
      var detokenizer = StreamingDetokenizer()
      var parser = NeedleToolCallParser()
      var generatedTokens = [EdgeToolsToken]()
      var confidence = EdgeToolsConfidenceState()
      var durationToFirstToken: Duration?

      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        let decoderLogits = try await self.decode(
          inputIds: MLTensor(tokenIds: [nextDecoderTokenId]),
          cachePosition: generatedTokens.count,
          encoderOutputs: encoderOutputs,
          cache: &cache
        )
        var stepLogits = decoderLogits.squeezingShape(at: 1)
        let processedLogits = try await processor?.process(logits: &stepLogits) ?? stepLogits
        let maskedLogits = applyBitmaskCoreML(logits: processedLogits, mask: matcher.bitmask())
        await confidence.addCoreML(logits: maskedLogits)

        let tokenId = try await sampler.sample(logits: maskedLogits)
        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)

        let tokenString = detokenizer.decode(tokenId: tokenId, using: self.tokenizer)
        let token = EdgeToolsToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(token)
        nextDecoderTokenId = tokenId
        guard matcher.accept(tokenId: token.id) else {
          throw NeedleCoreMLEngineError.grammarRejectedToken(token: token)
        }
        let rawToolCall = parser.accept(token: token)
        channel.emit(token: token)
        if let rawToolCall {
          channel.emit(toolCall: rawToolCall)
        }
        processor?.didSample(token: token)
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      var metadata = EdgeToolsMetadata()
      metadata.generationConfidence = confidence.mean
      metadata.perTokenConfidences = confidence.perTokenConfidences
      let response = self.tokenizer.decode(tokens: detokenizer.tokenIds)
      return EdgeToolsEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: generatedTokens.count,
          duration: generateStart.duration(to: self.clock.now) - finalDurationToFirstToken,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        response: response,
        metadata: metadata
      )
    }

    private func prefill(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      configuration: NeedleModelConfiguration,
      processor: inout (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?
    ) async throws -> (EncoderOutputs, EdgeToolsPrefillMetrics) {
      let promptTokens = try self.tokenizer.encode(text: prompt.formatted(tools: tools))
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw NeedleCoreMLEngineError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: configuration.encoderMaxLength
        )
      }

      let promptTensor = MLTensor(tokenIds: promptTokens)
      let paddedPromptTensor = MLTensor(
        tokenIds: promptTokens,
        paddingTo: configuration.encoderMaxLength,
        padTokenId: configuration.padTokenId
      )
      let prefillStart = self.clock.now
      processor?.prompt(promptTensor)
      let encoderOutputs = try await self.runEncoder(inputIds: paddedPromptTensor)
      let metrics = EdgeToolsPrefillMetrics(
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
        crossAttentionMask: crossAttentionMask,
        encoderProjectedK: encoderProjectedK,
        encoderProjectedV: encoderProjectedV
      )
    }

    private func decode(
      inputIds: MLTensor,
      cachePosition: Int,
      encoderOutputs: EncoderOutputs,
      cache: inout DecoderCache
    ) async throws -> MLTensor {
      let inputs = [
        TensorName.inputIds: inputIds,
        TensorName.cachePosition: MLTensor(shape: [1], scalars: [Int32(cachePosition)]),
        TensorName.selfAttentionMask: Self.selfAttentionMask(
          step: cachePosition,
          maxLength: self.configuration.encoderMaxLength
        ),
        TensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
        TensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
        TensorName.encoderProjectedV: encoderOutputs.encoderProjectedV,
        TensorName.keyCache: cache.key,
        TensorName.valueCache: cache.value
      ]
      let outputs = try await self.decoderModel.prediction(from: inputs)
      guard
        let logits = outputs[TensorName.logits],
        let updatedKey = outputs[TensorName.updatedKeyCache],
        let updatedValue = outputs[TensorName.updatedValueCache]
      else {
        throw NeedleCoreMLEngineError.missingModelOutputs
      }
      cache = DecoderCache(key: updatedKey, value: updatedValue)
      return logits
    }

  }

  // MARK: - Model Loading

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
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
      let aotCompiledURL =
        directory
        .appending(path: "compiled")
        .appending(path: Self.coreMLPlatform)
        .appending(path: "\(name).mlmodelc")
      if FileManager.default.fileExists(atPath: aotCompiledURL.path()) {
        return try await MLModel.load(contentsOf: aotCompiledURL, configuration: configuration)
      }

      #if os(watchOS)
        throw NeedleCoreMLEngineError.missingAOTModel(name: name, platform: Self.coreMLPlatform)
      #else
        let packageURL = directory.appending(path: "\(name).mlpackage")
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

    private static var coreMLPlatform: String {
      #if targetEnvironment(macCatalyst)
        "macCatalyst"
      #elseif os(macOS)
        "macOS"
      #elseif os(iOS)
        "iOS"
      #elseif os(watchOS)
        "watchOS"
      #elseif os(tvOS)
        "tvOS"
      #else
        "visionOS"
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

      init(model: sending MLModel) {
        self.model = model
      }

      func prediction(from inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        try await self.model.prediction(from: inputs)
      }
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

    public static func grammarRejectedToken(token: EdgeToolsToken) -> Self {
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

    public static func missingAOTModel(name: String, platform: String) -> Self {
      Self(
        message:
          "Could not find a precompiled CoreML model at compiled/\(platform)/\(name).mlmodelc. "
          + "watchOS requires models exported with --compile-platform \(platform)."
      )
    }
  }

  // MARK: - Helpers

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension MLTensor {
    fileprivate init(tokenIds: some Sequence<EdgeToolsToken.ID>) {
      let ids = tokenIds.map(Int32.init)
      self.init(shape: [1, ids.count], scalars: ids, scalarType: Int32.self)
    }

    fileprivate init(
      tokenIds: some Sequence<EdgeToolsToken.ID>,
      paddingTo length: Int,
      padTokenId: EdgeToolsToken.ID
    ) {
      var ids = tokenIds.map(Int32.init)
      ids.append(contentsOf: repeatElement(Int32(padTokenId), count: length - ids.count))
      self.init(shape: [1, length], scalars: ids, scalarType: Int32.self)
    }
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
    private static func selfAttentionMask(step: Int, maxLength: Int) -> MLTensor {
      var scalars = Array(repeating: Float16(-65500), count: maxLength)
      let allowedStart = max(0, maxLength - step - 1)
      for index in allowedStart..<maxLength {
        scalars[index] = 0
      }
      return MLTensor(shape: [1, 1, 1, maxLength], scalars: scalars)
    }
  }

  private enum ModelName {
    static let encoder = "encoder"
    static let decoder = "decoder"
  }

  private enum TensorName {
    static let inputIds = "input_ids"
    static let cachePosition = "cache_position"
    static let selfAttentionMask = "self_attention_mask"
    static let crossAttentionMask = "cross_attention_mask"
    static let encoderProjectedK = "encoder_projected_k"
    static let encoderProjectedV = "encoder_projected_v"
    static let keyCache = "key_cache"
    static let valueCache = "value_cache"
    static let updatedKeyCache = "updated_key_cache"
    static let updatedValueCache = "updated_value_cache"
    static let logits = "logits"
  }
#endif
