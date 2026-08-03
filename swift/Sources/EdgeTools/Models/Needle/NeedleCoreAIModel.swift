#if XGrammar
  import EdgeToolsXGrammar
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import _EdgeToolsFoundation

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIModel {
    public struct GenerateParameters: NeedleGenerateParameters {
      public static var `default`: Self { Self() }

      public var sampler: any EdgeToolsSampler<NDArray>
      public var processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?
      public var computeStream: ComputeStream?
      public var maxTokens: Int?
      public var toolCallRange: GrammarToolCallRange

      public init(
        sampler: any EdgeToolsSampler<NDArray> = CoreAIArgmaxSampler(),
        processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)? = nil,
        computeStream: ComputeStream? = nil,
        maxTokens: Int? = 1024,
        toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0)
      ) {
        self.sampler = sampler
        self.processor = processor
        self.computeStream = computeStream
        self.maxTokens = maxTokens
        self.toolCallRange = toolCallRange
      }
    }

    private struct EncoderOutputs {
      let crossAttentionMask: InferenceFunction.AsyncValue
      let encoderProjectedK: InferenceFunction.AsyncValue
      let encoderProjectedV: InferenceFunction.AsyncValue
    }

    private struct Generation {
      let encoderOutputs: EncoderOutputs
      let state: InferenceFunctionState
      let stream: ComputeStream?
      var position: Int
      var logits: NDArray
      var pendingTokenId: EdgeToolsToken.ID?
    }

    private let configuration: NeedleModelConfiguration
    private let encoderFunction: InferenceFunction
    private let decoderFunction: InferenceFunction
    private let clock = ContinuousClock()
    private var generation: Generation?

    public convenience init(
      modelDirectoryURL: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      specializationOptions: SpecializationOptions = .default,
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
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        configuration: configuration
      )
    }

    public init(
      encoderModel: AIModel,
      decoderModel: AIModel,
      configuration: NeedleModelConfiguration
    ) throws {
      self.configuration = configuration
      self.encoderFunction = try Self.loadFunction(named: FunctionName.main, from: encoderModel)
      self.decoderFunction = try Self.loadFunction(named: FunctionName.main, from: decoderModel)
    }
  }

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIModel {
    private nonisolated(nonsending) func runEncoder(
      inputIds: NDArray,
      stream: ComputeStream?
    ) async throws -> EncoderOutputs {
      let outputs = try await self.encoderFunction.invoke(
        inputs: [NeedleExportTensorName.inputIDs: InferenceFunction.AsyncValue(inputIds)],
        outputNames: [
          NeedleExportTensorName.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV
        ],
        on: stream
      )
      guard
        let crossAttentionMask = outputs[NeedleExportTensorName.crossAttentionMask],
        let encoderProjectedK = outputs[NeedleExportTensorName.encoderProjectedK],
        let encoderProjectedV = outputs[NeedleExportTensorName.encoderProjectedV]
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMask,
        encoderProjectedK: encoderProjectedK,
        encoderProjectedV: encoderProjectedV
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
      case .float16:
        let view = Span<UInt16>(viewing: logits.rawView().bytes)
        scalars = (0..<vocabularySize).map {
          Float(Float16(bitPattern: view[offset + $0]))
        }
      case .float32:
        let view = Span<Float>(viewing: logits.rawView().bytes)
        scalars = (0..<vocabularySize).map { view[offset + $0] }
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

  // MARK: - NeedleModel

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIModel: NeedleModel {
    public typealias Input = NeedleModelInput

    public var vocabularySize: Int { self.configuration.vocabularySize }

    public func input(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> EdgeToolsModelInput<NeedleModelInput> {
      let input = NeedleModelInput(
        prompt: prompt,
        tools: tools,
        tokenIds: try tokenizer.encode(text: prompt.formatted(tools: tools))
      )
      return EdgeToolsModelInput(value: input, tokenIds: input.tokenIds)
    }

    public nonisolated(nonsending) func prepare(
      input: NeedleModelInput,
      parameters: GenerateParameters
    ) async throws -> EdgeToolsModelPreparation {
      guard input.tokenIds.count <= self.configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: input.tokenIds.count,
          maximum: self.configuration.encoderMaxLength
        )
      }
      let promptArray = NDArray(tokenIds: input.tokenIds)
      let paddedPromptArray = NDArray(
        tokenIds: input.tokenIds,
        paddingTo: self.configuration.encoderMaxLength,
        padTokenId: self.configuration.padTokenId
      )
      let start = self.clock.now
      parameters.processor?.prompt(promptArray)
      let stream = parameters.computeStream
      let encoderOutputs = try await self.runEncoder(
        inputIds: paddedPromptArray,
        stream: parameters.computeStream
      )
      let state = try Self.decoderState(
        descriptor: self.decoderFunction.descriptor,
        configuration: self.configuration
      )
      let decoderLogits = try await self.decode(
        inputIds: NDArray(tokenIds: [self.configuration.decoderStartTokenId]),
        cachePosition: 0,
        encoderOutputs: encoderOutputs,
        state: state,
        stream: stream
      )
      let logits = try self.stepLogits(from: decoderLogits)
      self.generation = Generation(
        encoderOutputs: encoderOutputs,
        state: state,
        stream: stream,
        position: 1,
        logits: logits,
        pendingTokenId: nil
      )
      return EdgeToolsModelPreparation(
        metrics: EdgeToolsPrefillMetrics(
          tokens: input.tokenIds.count,
          duration: start.duration(to: self.clock.now)
        )
      )
    }

    public nonisolated(nonsending) func decode(
      bitmask: GrammarBitmask,
      parameters: GenerateParameters
    ) async throws -> EdgeToolsModelSample {
      guard var generation = self.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      if let pendingTokenId = generation.pendingTokenId {
        guard generation.position < self.configuration.decoderMaxLength else {
          throw EdgeToolsError.contextLengthExceeded(
            tokens: generation.position + 1,
            maximum: self.configuration.decoderMaxLength
          )
        }
        let decoderLogits = try await self.decode(
          inputIds: NDArray(tokenIds: [pendingTokenId]),
          cachePosition: generation.position,
          encoderOutputs: generation.encoderOutputs,
          state: generation.state,
          stream: generation.stream
        )
        generation.logits = try self.stepLogits(from: decoderLogits)
        generation.position += 1
      }
      try await parameters.processor?.process(logits: &generation.logits)
      var maskedLogits = generation.logits
      applyBitmask(logits: &maskedLogits, mask: bitmask)
      let confidence = try tokenConfidenceCoreAI(logits: maskedLogits)
      let tokenId = try await parameters.sampler.sample(logits: maskedLogits)
      parameters.processor?.didSample(tokenId: tokenId)
      generation.pendingTokenId = tokenId
      self.generation = generation
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public func resetGeneration() {
      self.generation = nil
    }
  }

  @available(anyAppleOS 27.0, *)
  public typealias NeedleCoreAIModelEngine = EdgeToolsModelEngine<NeedleCoreAIModel>

  @available(anyAppleOS 27.0, *)
  extension EdgeToolsModelEngine where Model == NeedleCoreAIModel {
    public init(
      modelDirectoryURL: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      specializationOptions: SpecializationOptions = SpecializationOptions(
        preferredComputeUnitKind: .neuralEngine
      ),
      modelCache: AIModelCache = .default,
      cachePolicy: AIModelCache.Policy = .default
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: modelDirectoryURL)
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let model = try await NeedleCoreAIModel(
        modelDirectoryURL: modelDirectoryURL,
        editConfiguration: editConfiguration,
        specializationOptions: specializationOptions,
        modelCache: modelCache,
        cachePolicy: cachePolicy
      )
      try self.init(model: model, tokenizer: tokenizer)
    }
  }

  // MARK: - Decoder

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIModel {
    private nonisolated(nonsending) func decode(
      inputIds: NDArray,
      cachePosition: Int,
      encoderOutputs: EncoderOutputs,
      state: InferenceFunctionState,
      stream: ComputeStream?
    ) async throws -> NDArray {
      let outputs = try await self.decoderFunction.invoke(
        inputs: [
          NeedleExportTensorName.inputIDs: InferenceFunction.AsyncValue(inputIds),
          NeedleExportTensorName.cachePosition: InferenceFunction.AsyncValue(
            NDArray(scalars: [Int32(cachePosition)], shape: [1])
          ),
          NeedleExportTensorName.selfAttentionMask: InferenceFunction.AsyncValue(
            Self.selfAttentionMask(
              step: cachePosition,
              maxLength: self.configuration.decoderMaxLength
            )
          ),
          NeedleExportTensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV: encoderOutputs.encoderProjectedV
        ],
        outputNames: [NeedleExportTensorName.logits],
        state: state,
        on: stream
      )
      guard let logits = outputs[NeedleExportTensorName.logits] else {
        throw EdgeToolsError.missingModelOutputs
      }
      guard let logitsArray = try await logits.ndArray else {
        throw EdgeToolsError.missingModelOutputs
      }
      return logitsArray
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
      public static let invalidStateShape = Self(rawValue: "invalid-state-shape")
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
  extension NeedleCoreAIModel {
    private static func decoderState(
      descriptor: InferenceFunctionDescriptor,
      configuration: NeedleModelConfiguration
    ) throws -> InferenceFunctionState {
      let names = (0..<configuration.decoderLayers).flatMap { layer in
        [
          NeedleExportTensorName.selfAttentionKeyCache(layer: layer),
          NeedleExportTensorName.selfAttentionValueCache(layer: layer)
        ]
      }
      guard Set(descriptor.stateNames) == Set(names) else {
        throw EdgeToolsCoreAIError(
          code: .missingModelStateDescriptors,
          message: "Expected CoreAI decoder states \(names), got \(descriptor.stateNames)."
        )
      }
      let values = try names.map { name in
        guard
          let descriptor = descriptor.stateDescriptor(of: name),
          case .ndArray(let arrayDescriptor) = descriptor
        else {
          throw EdgeToolsCoreAIError(
            code: .missingModelStateDescriptors,
            message: "CoreAI decoder did not expose an array state named \(name)."
          )
        }
        let expectedShape = [
          configuration.decoderMaxLength,
          configuration.kvHeads,
          configuration.attentionHeadDimensions
        ]
        guard arrayDescriptor.shape == expectedShape else {
          throw EdgeToolsCoreAIError(
            code: .invalidStateShape,
            message: "Expected state shape \(expectedShape), got \(arrayDescriptor.shape)."
          )
        }
        var array = NDArray(descriptor: arrayDescriptor)
        array.zero()
        return (name: name, array: array)
      }
      return InferenceFunctionState(values: values)
    }

    private static func selfAttentionMask(step: Int, maxLength: Int) -> NDArray {
      var mask = NDArray(shape: [1, 1, 1, maxLength], scalarType: .float16)
      var rawView = mask.mutableRawView()
      var values = MutableSpan<UInt16>(mutableBytes: rawView.mutableBytes)
      for index in 0..<maxLength {
        values[index] =
          index <= step ? 0 : Float16(NeedleNumerics.maskedAttentionScore).bitPattern
      }
      return mask
    }
  }

  private enum FunctionName {
    static let main = "main"
  }
#endif
