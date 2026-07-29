#if XGrammar
  import EdgeToolsXGrammar
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import _EdgeToolsFoundation

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIModel: Sendable {
    public typealias Prompt = NeedlePrompt

    fileprivate struct EncoderOutputs {
      let crossAttentionMask: InferenceFunction.AsyncValue
      let encoderProjectedK: InferenceFunction.AsyncValue
      let encoderProjectedV: InferenceFunction.AsyncValue
    }

    fileprivate struct DecoderCache {
      var key: InferenceFunction.AsyncValue
      var value: InferenceFunction.AsyncValue

      init(descriptor: InferenceFunctionDescriptor) throws {
        guard
          let keyDescriptor = descriptor.arrayDescriptor(
            for: NeedleExportTensorName.updatedKeyCache
          ),
          let valueDescriptor = descriptor.arrayDescriptor(
            for: NeedleExportTensorName.updatedValueCache
          )
        else {
          throw EdgeToolsError.missingModelOutputs
        }
        self.key = InferenceFunction.AsyncValue(NDArray(descriptor: keyDescriptor))
        self.value = InferenceFunction.AsyncValue(NDArray(descriptor: valueDescriptor))
      }
    }

    public struct ModelGenerationState {
      fileprivate let encoderOutputs: EncoderOutputs
      fileprivate var cache: DecoderCache
      fileprivate var position: Int
    }

    private let configuration: NeedleModelConfiguration
    private let encoderFunction: InferenceFunction
    private let decoderFunction: InferenceFunction
    private let clock = ContinuousClock()

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

  // MARK: - EdgeToolsModel

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIModel: EdgeToolsModel {
    public typealias Input = NeedleModelInput
    public typealias Logits = NDArray
    public typealias Assets = Void
    public typealias GenerateParameters = EdgeToolsCoreAIGenerateParameters
    public typealias GenerationState = EdgeToolsCoreAIGenerationState<ModelGenerationState>
    public typealias ToolCallParser = NeedleToolCallParser

    public var vocabularySize: Int { self.configuration.vocabularySize }

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.needle(tools: tools, range: range)
    }

    public func input(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> NeedleModelInput {
      NeedleModelInput(
        prompt: prompt,
        tools: tools,
        tokenIds: try tokenizer.encode(text: prompt.formatted(tools: tools))
      )
    }

    public func tokenIds(in input: NeedleModelInput) -> [EdgeToolsToken.ID] {
      input.tokenIds
    }

    public nonisolated(nonsending) func prepare(
      input: NeedleModelInput,
      parameters: GenerateParameters,
      assets: Void
    ) async throws -> EdgeToolsModelPreparation<NDArray, GenerationState> {
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
      let processor = parameters.processor
      processor?.prompt(promptArray)
      let stream = parameters.computeStream
      let encoderOutputs = try await self.runEncoder(
        inputIds: paddedPromptArray,
        stream: stream
      )
      var cache = try DecoderCache(descriptor: self.decoderFunction.descriptor)
      let decoderLogits = try await self.decode(
        inputIds: NDArray(tokenIds: [self.configuration.decoderStartTokenId]),
        cachePosition: 0,
        encoderOutputs: encoderOutputs,
        cache: &cache,
        stream: stream
      )
      let logits = try self.stepLogits(from: decoderLogits)
      let modelState = ModelGenerationState(
        encoderOutputs: encoderOutputs,
        cache: cache,
        position: 1
      )
      let state = EdgeToolsCoreAIGenerationState(
        modelState: modelState,
        parameters: parameters
      )
      return EdgeToolsModelPreparation(
        logits: logits,
        state: state,
        metrics: EdgeToolsPrefillMetrics(
          tokens: input.tokenIds.count,
          duration: start.duration(to: self.clock.now)
        )
      )
    }

    public nonisolated(nonsending) func decode(
      tokenId: EdgeToolsToken.ID,
      state: inout GenerationState,
      assets: Void
    ) async throws -> NDArray {
      let stream = state.computeStream
      let decoderLogits = try await self.decode(
        inputIds: NDArray(tokenIds: [tokenId]),
        cachePosition: state.modelState.position,
        encoderOutputs: state.modelState.encoderOutputs,
        cache: &state.modelState.cache,
        stream: stream
      )
      state.modelState.position += 1
      return try self.stepLogits(from: decoderLogits)
    }

    public nonisolated(nonsending) func sample(
      logits: inout NDArray,
      bitmask: GrammarBitmask,
      state: inout GenerationState
    ) async throws -> EdgeToolsModelSample {
      try await state.sample(logits: &logits, bitmask: bitmask)
    }

    public func didAccept(token: EdgeToolsToken, state: inout GenerationState) {
      state.didAccept(token: token)
    }
  }

  @available(anyAppleOS 27.0, *)
  public typealias NeedleCoreAIModelEngine = EdgeToolsModelEngine<NeedleCoreAIModel>

  @available(anyAppleOS 27.0, *)
  extension EdgeToolsModelEngine where Model == NeedleCoreAIModel {
    public init(
      model: sending NeedleCoreAIModel,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      try self.init(model: model, assets: (), tokenizer: tokenizer)
    }

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
      cache: inout DecoderCache,
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
              maxLength: self.configuration.encoderMaxLength
            )
          ),
          NeedleExportTensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV: encoderOutputs.encoderProjectedV,
          NeedleExportTensorName.keyCache: cache.key,
          NeedleExportTensorName.valueCache: cache.value
        ],
        outputNames: [
          NeedleExportTensorName.logits,
          NeedleExportTensorName.updatedKeyCache,
          NeedleExportTensorName.updatedValueCache
        ],
        on: stream
      )
      guard
        let logits = outputs[NeedleExportTensorName.logits],
        let key = outputs[NeedleExportTensorName.updatedKeyCache],
        let value = outputs[NeedleExportTensorName.updatedValueCache]
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      cache.key = key
      cache.value = value
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
  extension NeedleCoreAIModel {
    private static func selfAttentionMask(step: Int, maxLength: Int) -> NDArray {
      var mask = NDArray(shape: [1, 1, 1, maxLength], scalarType: .float16)
      let allowedStart = max(0, maxLength - step - 1)
      var rawView = mask.mutableRawView()
      var values = MutableSpan<UInt16>(mutableBytes: rawView.mutableBytes)
      for index in 0..<maxLength {
        values[index] = index >= allowedStart ? 0 : Float16(-65500).bitPattern
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
