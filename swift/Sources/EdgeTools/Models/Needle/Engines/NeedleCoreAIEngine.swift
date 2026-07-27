#if System
  import SystemPackage
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import Foundation
  import Atomics

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIEngine: EdgeToolsEngine {
    public typealias Prompt = NeedlePrompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self { Self() }

      public var sampler: any EdgeToolsSampler<NDArray>
      public var processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?
      private var _computeStream: @Sendable () -> ComputeStream?

      public var computeStream: ComputeStream? {
        self._computeStream()
      }

      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?

      public init(
        sampler: any EdgeToolsSampler<NDArray> = CoreAIArgmaxSampler(),
        processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)? = nil,
        computeStream: @autoclosure @escaping @Sendable () -> ComputeStream? = nil,
        constraint: XGRGenerationConstraint = .tools,
        maxTokens: Int? = 1024
      ) {
        self.sampler = sampler
        self.processor = processor
        self._computeStream = computeStream
        self.constraint = constraint
        self.maxTokens = maxTokens
      }
    }

    private struct State: ~Copyable {
      let grammarEngine: XGRCompiler
      let matcherPool: XGRToolCallMatcherPool
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
          let keyDescriptor = descriptor.arrayDescriptor(for: NeedleExportTensorName.updatedKeyCache),
          let valueDescriptor = descriptor.arrayDescriptor(for: NeedleExportTensorName.updatedValueCache)
        else {
          throw EdgeToolsError.missingModelOutputs
        }
        self.key = NDArray(descriptor: keyDescriptor)
        self.value = NDArray(descriptor: valueDescriptor)
      }
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderFunction: InferenceFunction
    private let decoderFunction: InferenceFunction
    private let tokenizer: any EdgeToolsXGRTokenizer
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
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: tokenizer,
        configuration: configuration
      )
    }

    #if System
      public convenience init(
        modelDirectoryPath: FilePath,
        editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
        specializationOptions: SpecializationOptions = SpecializationOptions(
          preferredComputeUnitKind: .neuralEngine
        ),
        modelCache: AIModelCache = .default,
        cachePolicy: AIModelCache.Policy = .default
      ) async throws {
        try await self.init(
          modelDirectoryURL: URL(filePath: modelDirectoryPath.string, directoryHint: .isDirectory),
          editConfiguration: editConfiguration,
          specializationOptions: specializationOptions,
          modelCache: modelCache,
          cachePolicy: cachePolicy
        )
      }
    #endif

    public init(
      encoderModel: AIModel,
      decoderModel: AIModel,
      tokenizer: sending any EdgeToolsXGRTokenizer,
      configuration: NeedleModelConfiguration
    ) throws {
      let grammarEngine = try XGRCompiler(
        tokenizerInfo: try tokenizer.tokenizerInfo(
          modelVocabularySize: configuration.vocabularySize
        )
      )
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          matcherPool: XGRToolCallMatcherPool()
        )
      )
      self.configuration = configuration
      self.encoderFunction = try Self.loadFunction(named: FunctionName.main, from: encoderModel)
      self.decoderFunction = try Self.loadFunction(named: FunctionName.main, from: decoderModel)
      self.tokenizer = tokenizer
    }
  }

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
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
        let toolsGrammar = try parameters.constraint.toolCallRange.map {
          try XGRGrammar.needle(tools: tools, range: $0)
        } ?? .universal
        let grammar = try parameters.constraint.grammar(using: toolsGrammar)
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            grammar: grammar,
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
      matcher: consuming XGRMatcher,
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
        tools: tools,
        configuration: configuration,
        processor: &processor,
        stream: stream
      )

      var cache = try DecoderCache(descriptor: self.decoderFunction.descriptor)
      var nextDecoderTokenId = configuration.decoderStartTokenId
      var loop = EdgeToolsGenerationLoop<NeedleToolCallParser>(
        matcher: consume matcher,
        tokenizer: self.tokenizer,
        channel: channel,
        isStopped: isStopped,
        maximumTokenCount: parameters.maxTokens,
        generateStart: generateStart
      )
      while let bitmask = try loop.nextBitmask() {
        let inputIDs = NDArray(tokenIds: [nextDecoderTokenId])
        let decoderLogits = try await self.decode(
          inputIds: inputIDs,
          cachePosition: loop.generatedTokenCount,
          encoderOutputs: encoderOutputs,
          cache: &cache,
          stream: stream
        )
        var logits = try self.stepLogits(from: decoderLogits)
        let processedLogits = try await processor?.process(logits: &logits) ?? logits
        var maskedLogits = processedLogits
        applyBitmaskCoreAI(logits: &maskedLogits, mask: bitmask)
        let confidence = try tokenConfidenceCoreAI(logits: maskedLogits)
        let tokenID = try await sampler.sample(logits: maskedLogits)
        let token = try loop.accept(tokenID: tokenID, confidence: confidence)
        nextDecoderTokenId = tokenID
        processor?.didSample(token: token)
      }

      return loop.finish(prefillMetrics: prefillMetrics)
    }

    private func prefill(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      configuration: NeedleModelConfiguration,
      processor: inout (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?,
      stream: ComputeStream?
    ) async throws -> (EncoderOutputs, EdgeToolsPrefillMetrics) {
      let promptTokens = try self.tokenizer.encode(text: prompt.formatted(tools: tools))
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
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
      var outputs = try await self.encoderFunction.run(inputs: [NeedleExportTensorName.inputIDs: inputIds])
      guard
        let crossAttentionMask = outputs.remove(NeedleExportTensorName.crossAttentionMask)?.ndArray,
        let encoderProjectedK = outputs.remove(NeedleExportTensorName.encoderProjectedK)?.ndArray,
        let encoderProjectedV = outputs.remove(NeedleExportTensorName.encoderProjectedV)?.ndArray
      else {
        throw EdgeToolsError.missingModelOutputs
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
        let crossMaskDescriptor = descriptor.arrayDescriptor(for: NeedleExportTensorName.crossAttentionMask),
        let projectedKDescriptor = descriptor.arrayDescriptor(for: NeedleExportTensorName.encoderProjectedK),
        let projectedVDescriptor = descriptor.arrayDescriptor(for: NeedleExportTensorName.encoderProjectedV)
      else {
        throw EdgeToolsError.missingModelOutputs
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
      outputViews.insert(&crossAttentionMask, for: NeedleExportTensorName.crossAttentionMask)
      outputViews.insert(&encoderProjectedK, for: NeedleExportTensorName.encoderProjectedK)
      outputViews.insert(&encoderProjectedV, for: NeedleExportTensorName.encoderProjectedV)

      let inputValues = [NeedleExportTensorName.inputIDs: InferenceFunction.AsyncValue(inputIds)]
      _ = try self.encoderFunction.encode(inputs: inputValues, outputViews: outputViews, to: stream)
      await stream.currentWorkCompleted()

      guard
        let crossAttentionMaskNDArray = try await crossAttentionMask.ndArray,
        let encoderProjectedKNDArray = try await encoderProjectedK.ndArray,
        let encoderProjectedVNDArray = try await encoderProjectedV.ndArray
      else {
        throw EdgeToolsError.missingModelOutputs
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
        throw EdgeToolsCoreAIError(
          code: .unsupportedLogitsScalarType,
          message: "Unsupported logits scalar type: \(logits.scalarType)."
        )
      }
      return NDArray(scalars: scalars, shape: [1, scalars.count])
    }

    private static func decodeConfiguration(
      from directory: URL
    ) throws -> NeedleModelConfiguration {
      guard let config = try NeedleModelConfiguration.decode(in: directory) else {
        throw EdgeToolsError.failedToLoadConfiguration
      }
      return config
    }

    private static func loadFunction(
      named name: String,
      from model: AIModel
    ) throws -> InferenceFunction {
      guard let function = try model.loadFunction(named: name) else {
        throw EdgeToolsCoreAIError(
          code: .failedToLoadFunction,
          message: "Could not load CoreAI function named \(name)."
        )
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
          NeedleExportTensorName.inputIDs: inputIds,
          NeedleExportTensorName.cachePosition: NDArray(scalars: [Int32(cachePosition)], shape: [1]),
          NeedleExportTensorName.selfAttentionMask: Self.selfAttentionMask(
            step: cachePosition,
            maxLength: self.configuration.encoderMaxLength
          ),
          NeedleExportTensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV: encoderOutputs.encoderProjectedV,
          NeedleExportTensorName.keyCache: cache.key,
          NeedleExportTensorName.valueCache: cache.value
        ]
      )
      guard
        let logits = outputs.remove(NeedleExportTensorName.logits)?.ndArray,
        let updatedKey = outputs.remove(NeedleExportTensorName.updatedKeyCache)?.ndArray,
        let updatedValue = outputs.remove(NeedleExportTensorName.updatedValueCache)?.ndArray
      else {
        throw EdgeToolsError.missingModelOutputs
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
          NeedleExportTensorName.inputIDs: InferenceFunction.AsyncValue(inputIds),
          NeedleExportTensorName.cachePosition: InferenceFunction.AsyncValue(
            NDArray(scalars: [Int32(cachePosition)], shape: [1])
          ),
          NeedleExportTensorName.selfAttentionMask: InferenceFunction.AsyncValue(
            Self.selfAttentionMask(
              step: cachePosition,
              maxLength: self.configuration.encoderMaxLength
            )
          ),
          NeedleExportTensorName.crossAttentionMask: InferenceFunction.AsyncValue(
            encoderOutputs.crossAttentionMask
          ),
          NeedleExportTensorName.encoderProjectedK: InferenceFunction.AsyncValue(
            encoderOutputs.encoderProjectedK
          ),
          NeedleExportTensorName.encoderProjectedV: InferenceFunction.AsyncValue(
            encoderOutputs.encoderProjectedV
          ),
          NeedleExportTensorName.keyCache: InferenceFunction.AsyncValue(cache.key),
          NeedleExportTensorName.valueCache: InferenceFunction.AsyncValue(cache.value)
        ],
        to: stream
      )
      await stream.currentWorkCompleted()

      guard
        let logits = try await outputs[NeedleExportTensorName.logits]?.ndArray,
        let key = try await outputs[NeedleExportTensorName.updatedKeyCache]?.ndArray,
        let value = try await outputs[NeedleExportTensorName.updatedValueCache]?.ndArray
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      cache.key = key
      cache.value = value
      return logits
    }
  }

  // MARK: - EdgeToolsCoreAIError

  @available(anyAppleOS 27.0, *)
  public struct EdgeToolsCoreAIError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let failedToLoadFunction = Self(rawValue: "failed-to-load-function")
      public static let missingModelStateDescriptors = Self(
        rawValue: "missing-model-state-descriptors"
      )
      public static let unsupportedLogitsScalarType = Self(
        rawValue: "unsupported-logits-scalar-type"
      )
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }

  }

  // MARK: - Helpers

  @available(anyAppleOS 27.0, *)
  extension NDArray {
    fileprivate init(descriptor: InferenceValue.Descriptor) throws {
      guard case .ndArray(let descriptor) = descriptor else {
        throw EdgeToolsCoreAIError(
          code: .missingModelStateDescriptors,
          message: "CoreAI model did not return expected state descriptors."
        )
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

#endif
