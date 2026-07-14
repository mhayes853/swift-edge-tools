#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import Foundation
  import Atomics

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIEngine: EdgeToolsEngine {
    public typealias Prompt = NeedlePrompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self { Self() }

      private var _sampler: @Sendable () -> any EdgeToolsSampler<NDArray>
      private var _processor: @Sendable () -> (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?
      private var _computeStream: @Sendable () -> ComputeStream?

      public var sampler: any EdgeToolsSampler<NDArray> {
        self._sampler()
      }

      public var processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)? {
        self._processor()
      }

      public var computeStream: ComputeStream? {
        self._computeStream()
      }

      public var toolCallRange: GrammarToolCallRange
      public var maxTokens: Int?

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any EdgeToolsSampler<NDArray> =
          CoreAIArgmaxSampler(),
        processor:
          @autoclosure @escaping @Sendable () -> (any EdgeToolsLogitsProcessor<NDArray, NDArray>)? =
          nil,
        computeStream: @autoclosure @escaping @Sendable () -> ComputeStream? = nil,
        toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
        maxTokens: Int? = 1024
      ) {
        self._sampler = sampler
        self._processor = processor
        self._computeStream = computeStream
        self.toolCallRange = toolCallRange
        self.maxTokens = maxTokens
      }
    }

    private struct State: ~Copyable {
      let grammarEngine: XGrammarCompiler
      let matcherPool: NeedleGrammarMatcherPool
    }

    private struct EncoderOutputs {
      let crossAttentionMask: NDArray
      let encoderProjectedK: NDArray
      let encoderProjectedV: NDArray
    }

    private struct DecoderCache {
      var key: NDArray
      var value: NDArray

      init(descriptor: InferenceFunctionDescriptor) throws {
        guard
          let keyDescriptor = descriptor.arrayDescriptor(for: TensorName.updatedKeyCache),
          let valueDescriptor = descriptor.arrayDescriptor(for: TensorName.updatedValueCache)
        else {
          throw NeedleCoreAIEngineError.missingModelOutputs
        }
        self.key = NDArray(descriptor: keyDescriptor)
        self.value = NDArray(descriptor: valueDescriptor)
      }
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderFunction: InferenceFunction
    private let decoderFunction: InferenceFunction
    private let tokenizer: Lock<any EdgeToolsTokenizer & ~Copyable>
    private let clock = ContinuousClock()

    public convenience init(
      modelDirectoryURL: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      specializationOptions: SpecializationOptions = SpecializationOptions(
        preferredComputeUnitKind: .neuralEngine
      ),
      modelCache: AIModelCache = .default,
      cachePolicy: AIModelCache.Policy = .default
    ) async throws {
      var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
      editConfiguration(&configuration)

      async let encoderModel = Self.loadModel(
        named: "encoder",
        from: modelDirectoryURL,
        options: specializationOptions,
        cache: modelCache,
        cachePolicy: cachePolicy
      )
      async let decoderModel = Self.loadModel(
        named: "decoder",
        from: modelDirectoryURL,
        options: specializationOptions,
        cache: modelCache,
        cachePolicy: cachePolicy
      )
      let tokenizer = try await loadEdgeToolsTokenizer(from: modelDirectoryURL)
      let lockedTokenizer = Lock(consume tokenizer)
      let grammarEngine = lockedTokenizer.withLock { XGrammarCompiler.needle(erasedTokenizer: $0) }
      guard let grammarEngine else {
        throw XGrammarError(message: "Needle requires a tokenizer with an EOS token.")
      }
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: consume lockedTokenizer,
        configuration: configuration,
        grammarEngine: consume grammarEngine
      )
    }

    public convenience init<Tokenizer: EdgeToolsTokenizer & ~Copyable>(
      encoderModel: AIModel,
      decoderModel: AIModel,
      tokenizer: consuming sending Tokenizer,
      configuration: NeedleModelConfiguration,
      grammarEngine: consuming sending XGrammarCompiler
    ) throws {
      try self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: Lock(consume tokenizer),
        configuration: configuration,
        grammarEngine: consume grammarEngine
      )
    }

    private init(
      encoderModel: AIModel,
      decoderModel: AIModel,
      tokenizer: consuming sending Lock<any EdgeToolsTokenizer & ~Copyable>,
      configuration: NeedleModelConfiguration,
      grammarEngine: consuming sending XGrammarCompiler
    ) throws {
      self.state = Lock(
        State(grammarEngine: consume grammarEngine, matcherPool: NeedleGrammarMatcherPool())
      )
      self.configuration = configuration
      self.encoderFunction = try Self.loadFunction(named: FunctionName.main, from: encoderModel)
      self.decoderFunction = try Self.loadFunction(named: FunctionName.main, from: decoderModel)
      self.tokenizer = consume tokenizer
    }
  }

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
    public func tokenize(prompt: NeedlePrompt) async throws -> [EdgeToolsToken] {
      try self.tokenizer.withBorrowedLock { try prompt.tokenized(using: $0) }
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
      channel: EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        let range = parameters.toolCallRange
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            tools: prompt.tools.map(\.definition),
            range: range,
            compilingWith: state.grammarEngine
          )
          matcher.reset()
          return matcher
        }
        return try await self.generate(
          prompt: prompt,
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
      let stream = parameters.computeStream
      let generateStart = self.clock.now
      let (encoderOutputs, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        configuration: configuration,
        processor: &processor,
        stream: stream
      )

      var cache = try DecoderCache(descriptor: self.decoderFunction.descriptor)
      var nextDecoderTokenId = configuration.decoderStartTokenId
      var detokenizer = StreamingDetokenizer()
      var parser = NeedleToolCallParser()
      var generatedTokens = [EdgeToolsToken]()
      var confidence = EdgeToolsConfidenceState()
      var durationToFirstToken: Duration?

      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.withBorrowedLock({ $0.eosTokenId })
      {
        try Task.checkCancellation()
        let inputIds = NDArray(tokenIds: [nextDecoderTokenId])
        let decoderLogits = try await self.decode(
          inputIds: inputIds,
          cachePosition: generatedTokens.count,
          encoderOutputs: encoderOutputs,
          cache: &cache,
          stream: stream
        )
        var logits = try self.stepLogits(from: decoderLogits)
        let processedLogits = try await processor?.process(logits: &logits) ?? logits
        var maskedLogits = processedLogits
        applyBitmaskCoreAI(logits: &maskedLogits, mask: matcher.bitmask())
        try confidence.addCoreAI(logits: maskedLogits)

        let tokenId = try await sampler.sample(logits: maskedLogits)
        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)

        let tokenString = self.tokenizer.withBorrowedLock {
          detokenizer.decode(tokenId: tokenId, using: $0)
        }
        let token = EdgeToolsToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(token)
        nextDecoderTokenId = tokenId
        guard matcher.accept(tokenId: token.id) else {
          throw NeedleCoreAIEngineError.grammarRejectedToken(token: token)
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
      return EdgeToolsEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
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
      processor: inout (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?,
      stream: ComputeStream?
    ) async throws -> (EncoderOutputs, EdgeToolsPrefillMetrics) {
      let promptTokens = try self.tokenizer.withBorrowedLock {
        try $0.encode(text: prompt.formatted())
      }
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw NeedleCoreAIEngineError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: configuration.encoderMaxLength
        )
      }

      let promptArray = NDArray(tokenIds: promptTokens)
      let paddedPromptArray = NDArray(
        tokenIds: promptTokens,
        paddingTo: configuration.encoderMaxLength,
        padTokenId: configuration.padTokenId
      )
      let prefillStart = self.clock.now
      processor?.prompt(promptArray)
      let encoderOutputs = try await self.runEncoder(inputIds: paddedPromptArray, stream: stream)
      let metrics = EdgeToolsPrefillMetrics(
        tokens: promptTokens.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      return (encoderOutputs, metrics)
    }

    private func runEncoder(
      inputIds: NDArray,
      stream: ComputeStream?
    ) async throws -> EncoderOutputs {
      if let stream {
        return try await self.runEncoderOnStream(inputIds: inputIds, stream: stream)
      }
      var outputs = try await self.encoderFunction.run(inputs: [TensorName.inputIds: inputIds])
      guard
        let crossAttentionMask = outputs.remove(TensorName.crossAttentionMask)?.ndArray,
        let encoderProjectedK = outputs.remove(TensorName.encoderProjectedK)?.ndArray,
        let encoderProjectedV = outputs.remove(TensorName.encoderProjectedV)?.ndArray
      else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMask,
        encoderProjectedK: encoderProjectedK,
        encoderProjectedV: encoderProjectedV
      )
    }

    private func runEncoderOnStream(
      inputIds: NDArray,
      stream: ComputeStream
    ) async throws -> EncoderOutputs {
      let descriptor = self.encoderFunction.descriptor
      guard
        let crossMaskDescriptor = descriptor.arrayDescriptor(for: TensorName.crossAttentionMask),
        let projectedKDescriptor = descriptor.arrayDescriptor(for: TensorName.encoderProjectedK),
        let projectedVDescriptor = descriptor.arrayDescriptor(for: TensorName.encoderProjectedV)
      else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }

      let crossShape = [1, 1, 1, self.configuration.encoderMaxLength]
      var crossAttentionMask = InferenceFunction.AsyncMutableValue(
        NDArray(descriptor: crossMaskDescriptor.resolvingDynamicDimensions(crossShape))
      )

      let kvShape = [
        self.configuration.decoderLayers,
        1,
        self.configuration.kvHeads,
        self.configuration.encoderMaxLength,
        self.configuration.attentionHeadDimensions
      ]
      var encoderProjectedK = InferenceFunction.AsyncMutableValue(
        NDArray(descriptor: projectedKDescriptor.resolvingDynamicDimensions(kvShape))
      )
      var encoderProjectedV = InferenceFunction.AsyncMutableValue(
        NDArray(descriptor: projectedVDescriptor.resolvingDynamicDimensions(kvShape))
      )

      var outputViews = InferenceFunction.AsyncMutableViews()
      outputViews.insert(&crossAttentionMask, for: TensorName.crossAttentionMask)
      outputViews.insert(&encoderProjectedK, for: TensorName.encoderProjectedK)
      outputViews.insert(&encoderProjectedV, for: TensorName.encoderProjectedV)

      let inputValues = [TensorName.inputIds: InferenceFunction.AsyncValue(inputIds)]
      _ = try self.encoderFunction.encode(inputs: inputValues, outputViews: outputViews, to: stream)
      await stream.currentWorkCompleted()

      guard
        let crossAttentionMaskNDArray = try await crossAttentionMask.ndArray,
        let encoderProjectedKNDArray = try await encoderProjectedK.ndArray,
        let encoderProjectedVNDArray = try await encoderProjectedV.ndArray
      else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMaskNDArray,
        encoderProjectedK: encoderProjectedKNDArray,
        encoderProjectedV: encoderProjectedVNDArray
      )
    }

    private func stepLogits(from logits: NDArray) throws -> NDArray {
      let stepIndex = logits.shape[1] - 1
      let vocabularySize = logits.shape[2]
      let offset = stepIndex * vocabularySize
      let scalars: [Float]
      switch logits.scalarType {
      case .bfloat16:
        let view = Span<UInt16>(viewing: logits.rawView().bytes)
        scalars = (0..<vocabularySize).map { Float(bfloat16Bits: view[offset + $0]) }
      default:
        throw NeedleCoreAIEngineError.unsupportedLogitsScalarType(logits.scalarType)
      }
      return NDArray(scalars: scalars, shape: [1, scalars.count])
    }

    private static func decodeConfiguration(
      from directory: URL
    ) throws -> NeedleModelConfiguration {
      guard let config = try NeedleModelConfiguration.decode(in: directory) else {
        throw NeedleCoreAIEngineError.failedToLoadConfiguration
      }
      return config
    }

    private static func loadFunction(
      named name: String,
      from model: AIModel
    ) throws -> InferenceFunction {
      guard let function = try model.loadFunction(named: name) else {
        throw NeedleCoreAIEngineError.failedToLoadFunction(name: name)
      }
      return function
    }

    private static func loadModel(
      named name: String,
      from directory: URL,
      options: SpecializationOptions,
      cache: AIModelCache,
      cachePolicy: AIModelCache.Policy
    ) async throws -> AIModel {
      do {
        let compiledModelURL = directory.appending(
          path: "\(name).\(AIModel.deviceArchitectureName).aimodelc"
        )
        return try await Self.loadCachedModel(
          contentsOf: compiledModelURL,
          options: options,
          cache: cache,
          cachePolicy: cachePolicy
        )
      } catch {
        let modelURL = directory.appending(path: "\(name).aimodel")
        return try await Self.loadCachedModel(
          contentsOf: modelURL,
          options: options,
          cache: cache,
          cachePolicy: cachePolicy
        )
      }
    }

    private static func loadCachedModel(
      contentsOf modelURL: URL,
      options: SpecializationOptions,
      cache: AIModelCache,
      cachePolicy: AIModelCache.Policy
    ) async throws -> AIModel {
      if let model = try cache.model(for: modelURL, options: options) {
        return model
      }
      return try await AIModel.specialize(
        contentsOf: modelURL,
        options: options,
        cache: cache,
        cachePolicy: cachePolicy
      )
    }
  }

  // MARK: - Decoder

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
    private func decode(
      inputIds: NDArray,
      cachePosition: Int,
      encoderOutputs: EncoderOutputs,
      cache: inout DecoderCache,
      stream: ComputeStream?
    ) async throws -> NDArray {
      if let stream {
        return try await self.decodeOnStream(
          inputIds: inputIds,
          cachePosition: cachePosition,
          encoderOutputs: encoderOutputs,
          cache: &cache,
          stream: stream
        )
      }
      var outputs = try await self.decoderFunction.run(
        inputs: [
          TensorName.inputIds: inputIds,
          TensorName.cachePosition: NDArray(scalars: [Int32(cachePosition)], shape: [1]),
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
      )
      guard
        let logits = outputs.remove(TensorName.logits)?.ndArray,
        let updatedKey = outputs.remove(TensorName.updatedKeyCache)?.ndArray,
        let updatedValue = outputs.remove(TensorName.updatedValueCache)?.ndArray
      else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }
      cache.key = updatedKey
      cache.value = updatedValue
      return logits
    }

    private func decodeOnStream(
      inputIds: NDArray,
      cachePosition: Int,
      encoderOutputs: EncoderOutputs,
      cache: inout DecoderCache,
      stream: ComputeStream
    ) async throws -> NDArray {
      let outputs = try self.decoderFunction.encode(
        inputs: [
          TensorName.inputIds: InferenceFunction.AsyncValue(inputIds),
          TensorName.cachePosition: InferenceFunction.AsyncValue(
            NDArray(scalars: [Int32(cachePosition)], shape: [1])
          ),
          TensorName.selfAttentionMask: InferenceFunction.AsyncValue(
            Self.selfAttentionMask(
              step: cachePosition,
              maxLength: self.configuration.encoderMaxLength
            )
          ),
          TensorName.crossAttentionMask: InferenceFunction.AsyncValue(
            encoderOutputs.crossAttentionMask
          ),
          TensorName.encoderProjectedK: InferenceFunction.AsyncValue(
            encoderOutputs.encoderProjectedK
          ),
          TensorName.encoderProjectedV: InferenceFunction.AsyncValue(
            encoderOutputs.encoderProjectedV
          ),
          TensorName.keyCache: InferenceFunction.AsyncValue(cache.key),
          TensorName.valueCache: InferenceFunction.AsyncValue(cache.value)
        ],
        to: stream
      )
      await stream.currentWorkCompleted()

      guard
        let logits = try await outputs[TensorName.logits]?.ndArray,
        let key = try await outputs[TensorName.updatedKeyCache]?.ndArray,
        let value = try await outputs[TensorName.updatedValueCache]?.ndArray
      else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }
      cache.key = key
      cache.value = value
      return logits
    }
  }

  // MARK: - NeedleCoreAIEngineError

  @available(anyAppleOS 27.0, *)
  public struct NeedleCoreAIEngineError: Hashable, Error {
    public let message: String

    public static let failedToLoadConfiguration = Self(
      message: "Could not load model configuration."
    )

    public static func failedToLoadFunction(name: String) -> Self {
      Self(message: "Could not load CoreAI function named \(name).")
    }

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
      message: "CoreAI model did not return expected outputs."
    )

    public static let missingModelStateDescriptors = Self(
      message: "CoreAI model did not return expected state descriptors."
    )

    public static func unsupportedLogitsScalarType(_ scalarType: NDArray.ScalarType) -> Self {
      Self(message: "Unsupported logits scalar type: \(scalarType).")
    }
  }

  // MARK: - Helpers

  @available(anyAppleOS 27.0, *)
  extension NDArray {
    fileprivate init(descriptor: InferenceValue.Descriptor) throws {
      guard case .ndArray(let descriptor) = descriptor else {
        throw NeedleCoreAIEngineError.missingModelStateDescriptors
      }
      self.init(descriptor: descriptor)
    }

    fileprivate init(tokenIds: some Sequence<EdgeToolsToken.ID>) {
      let ids = tokenIds.map(Int32.init)
      self.init(scalars: ids, shape: [1, ids.count])
    }

    fileprivate init(
      tokenIds: some Sequence<EdgeToolsToken.ID>,
      paddingTo length: Int,
      padTokenId: EdgeToolsToken.ID
    ) {
      var ids = tokenIds.map(Int32.init)
      ids.append(contentsOf: repeatElement(Int32(padTokenId), count: length - ids.count))
      self.init(scalars: ids, shape: [1, length])
    }

    fileprivate mutating func zero() {
      var view = self.mutableRawView()
      let count = view.mutableBytes.byteCount
      var span = MutableSpan<UInt8>(mutableBytes: view.mutableBytes)
      for index in 0..<count {
        span[index] = 0
      }
    }
  }

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
    private static func selfAttentionMask(step: Int, maxLength: Int) -> NDArray {
      var mask = NDArray(shape: [1, 1, 1, maxLength], scalarType: .bfloat16)
      let allowedStart = max(0, maxLength - step - 1)
      var rawView = mask.mutableRawView()
      var values = MutableSpan<UInt16>(mutableBytes: rawView.mutableBytes)
      for index in 0..<maxLength {
        values[index] = index >= allowedStart ? 0 : 0xC77F
      }
      return mask
    }
  }

  @available(anyAppleOS 27.0, *)
  extension InferenceFunctionDescriptor {
    fileprivate func arrayDescriptor(for name: String) -> NDArrayDescriptor? {
      guard case .ndArray(let arrayDescriptor) = self.outputDescriptor(of: name) else { return nil }
      return arrayDescriptor
    }
  }

  private enum FunctionName {
    static let main = "main"
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
