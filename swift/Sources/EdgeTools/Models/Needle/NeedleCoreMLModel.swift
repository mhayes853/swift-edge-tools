#if XGrammar
  import EdgeToolsXGrammar
#endif

#if CoreML && canImport(CoreML)
  @preconcurrency import CoreML
  import _EdgeToolsFoundation

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public final class NeedleCoreMLModel {
    public struct GenerateParameters: NeedleGenerateParameters {
      public static var `default`: Self { Self() }

      public var sampler: any EdgeToolsSampler<MLTensor>
      public var processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?
      public var maxTokens: Int?
      public var toolCallRange: GrammarToolCallRange

      public init(
        sampler: any EdgeToolsSampler<MLTensor> = CoreMLArgmaxSampler(),
        processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)? = nil,
        maxTokens: Int? = 1024,
        toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0)
      ) {
        self.sampler = sampler
        self.processor = processor
        self.maxTokens = maxTokens
        self.toolCallRange = toolCallRange
      }
    }

    fileprivate struct EncoderOutputs {
      let crossAttentionMask: MLTensor
      let encoderProjectedK: MLTensor
      let encoderProjectedV: MLTensor
    }

    private struct Generation {
      let crossAttentionMask: MLTensor
      let state: MLState
      var position: Int
      var logits: MLTensor
      var pendingTokenId: EdgeToolsToken.ID?
    }

    private let configuration: NeedleModelConfiguration
    private let encoderModel: EncoderModelActor
    private let decoderModel: DecoderModelActor
    private let clock = ContinuousClock()
    private var generation: Generation?

    public convenience init(
      modelDirectoryURL: URL,
      modelConfiguration: MLModelConfiguration,
      editModelConfiguration: (inout MLModelConfiguration) -> Void = { _ in },
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
    ) async throws {
      var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
      editConfiguration(&configuration)
      var modelConfiguration = modelConfiguration
      editModelConfiguration(&modelConfiguration)

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
      await self.init(
        encoderModel: try encoderModel,
        decoderModel: try decoderModel,
        configuration: configuration
      )
    }

    public init(
      encoderModel: sending MLModel,
      decoderModel: sending MLModel,
      configuration: NeedleModelConfiguration
    ) {
      self.configuration = configuration
      self.encoderModel = EncoderModelActor(model: encoderModel)
      self.decoderModel = DecoderModelActor(model: decoderModel)
    }

    private nonisolated(nonsending) func runEncoder(
      inputIds: MLTensor
    ) async throws -> EncoderOutputs {
      var outputs = try await self.encoderModel.prediction(from: [
        NeedleExportTensorName.inputIDs: inputIds
      ])
      guard
        let crossAttentionMask = outputs.removeValue(
          forKey: NeedleExportTensorName.crossAttentionMask
        ),
        let encoderProjectedK = outputs.removeValue(
          forKey: NeedleExportTensorName.encoderProjectedK
        ),
        let encoderProjectedV = outputs.removeValue(
          forKey: NeedleExportTensorName.encoderProjectedV
        )
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMask,
        encoderProjectedK: encoderProjectedK,
        encoderProjectedV: encoderProjectedV
      )
    }

    private nonisolated(nonsending) func decode(
      inputIds: MLTensor,
      cachePosition: Int,
      crossAttentionMask: MLTensor,
      state: MLState
    ) async throws -> MLTensor {
      let inputs = [
        NeedleExportTensorName.inputIDs: inputIds,
        NeedleExportTensorName.cachePosition: MLTensor(shape: [1], scalars: [Int32(cachePosition)]),
        NeedleExportTensorName.selfAttentionMask: Self.selfAttentionMask(
          step: cachePosition,
          maxLength: self.configuration.decoderMaxLength
        ),
        NeedleExportTensorName.crossAttentionMask: crossAttentionMask
      ]
      let outputs = try await self.decoderModel.prediction(from: inputs, using: state)
      guard let logits = outputs[NeedleExportTensorName.logits] else {
        throw EdgeToolsError.missingModelOutputs
      }
      return logits
    }

  }

  // MARK: - NeedleModel

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLModel: NeedleModel {
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
      let promptTensor = MLTensor(tokenIds: input.tokenIds)
      let paddedPromptTensor = MLTensor(
        tokenIds: input.tokenIds,
        paddingTo: self.configuration.encoderMaxLength,
        padTokenId: self.configuration.padTokenId
      )
      let start = self.clock.now
      parameters.processor?.prompt(promptTensor)
      let encoderOutputs = try await self.runEncoder(inputIds: paddedPromptTensor)
      let state = try await self.decoderModel.makeState(
        encoderProjectedK: encoderOutputs.encoderProjectedK,
        encoderProjectedV: encoderOutputs.encoderProjectedV,
        configuration: self.configuration
      )
      let logits = try await self.decode(
        inputIds: MLTensor(tokenIds: [self.configuration.decoderStartTokenId]),
        cachePosition: 0,
        crossAttentionMask: encoderOutputs.crossAttentionMask,
        state: state
      )
      self.generation = Generation(
        crossAttentionMask: encoderOutputs.crossAttentionMask,
        state: state,
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
        generation.logits = try await self.decode(
          inputIds: MLTensor(tokenIds: [pendingTokenId]),
          cachePosition: generation.position,
          crossAttentionMask: generation.crossAttentionMask,
          state: generation.state
        )
        generation.position += 1
      }
      var stepLogits = generation.logits.squeezingShape(at: 1)
      try await parameters.processor?.process(logits: &stepLogits)
      let maskedLogits = applyBitmaskCoreML(logits: stepLogits, mask: bitmask)
      let confidence = await tokenConfidenceCoreML(logits: maskedLogits)
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

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public typealias NeedleCoreMLModelEngine = EdgeToolsModelEngine<NeedleCoreMLModel>

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension EdgeToolsModelEngine where Model == NeedleCoreMLModel {
    public init(
      modelDirectoryURL: URL,
      modelConfiguration: MLModelConfiguration,
      editModelConfiguration: (inout MLModelConfiguration) -> Void = { _ in },
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: modelDirectoryURL)
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let model = try await NeedleCoreMLModel(
        modelDirectoryURL: modelDirectoryURL,
        modelConfiguration: modelConfiguration,
        editModelConfiguration: editModelConfiguration,
        editConfiguration: editConfiguration
      )
      try self.init(model: model, tokenizer: tokenizer)
    }
  }

  // MARK: - Model Loading

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLModel {
    private static func decodeConfiguration(
      from directory: URL
    ) throws -> NeedleModelConfiguration {
      guard let config = try NeedleModelConfiguration.decode(in: directory) else {
        throw EdgeToolsError.failedToLoadConfiguration
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
        throw EdgeToolsCoreMLError(
          code: .missingAOTModel,
          message:
            "Could not find a precompiled CoreML model at compiled/\(Self.coreMLPlatform)/\(name).mlmodelc. "
            + "watchOS requires models exported with --compile-platform \(Self.coreMLPlatform)."
        )
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
  extension NeedleCoreMLModel {
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

      func prediction(
        from inputs: [String: MLTensor],
        using state: MLState
      ) async throws -> [String: MLTensor] {
        try await self.model.prediction(from: inputs, using: state)
      }

      func makeState(
        encoderProjectedK: MLTensor,
        encoderProjectedV: MLTensor,
        configuration: NeedleModelConfiguration
      ) async throws -> MLState {
        async let projectedK = encoderProjectedK.shapedArray(of: Float16.self)
        async let projectedV = encoderProjectedV.shapedArray(of: Float16.self)
        let (keys, values) = await (projectedK, projectedV)
        let state = self.model.makeState()
        for layer in 0..<configuration.decoderLayers {
          try Self.zeroSelfAttention(
            state: state,
            name: NeedleExportTensorName.selfAttentionKeyCache(layer: layer),
            configuration: configuration
          )
          try Self.zeroSelfAttention(
            state: state,
            name: NeedleExportTensorName.selfAttentionValueCache(layer: layer),
            configuration: configuration
          )
          try Self.initializeCrossAttention(
            state: state,
            name: NeedleExportTensorName.crossAttentionKeyCache(layer: layer),
            projected: keys,
            layer: layer,
            configuration: configuration
          )
          try Self.initializeCrossAttention(
            state: state,
            name: NeedleExportTensorName.crossAttentionValueCache(layer: layer),
            projected: values,
            layer: layer,
            configuration: configuration
          )
        }
        return state
      }

      private static func zeroSelfAttention(
        state: MLState,
        name: String,
        configuration: NeedleModelConfiguration
      ) throws {
        try state.withMultiArray(for: name) { array in
          let stateShape = array.shape.map(\.intValue)
          let expectedStateShape = [
            configuration.decoderMaxLength,
            configuration.kvHeads,
            configuration.attentionHeadDimensions
          ]
          guard stateShape == expectedStateShape else {
            throw EdgeToolsCoreMLError(
              code: .invalidStateShape,
              message: "Expected state shape \(expectedStateShape), got \(stateShape)."
            )
          }
          array.withUnsafeMutableBufferPointer(ofType: Float16.self) { values, _ in
            for index in values.indices {
              values[index] = 0
            }
          }
        }
      }

      private static func initializeCrossAttention(
        state: MLState,
        name: String,
        projected: MLShapedArray<Float16>,
        layer: Int,
        configuration: NeedleModelConfiguration
      ) throws {
        let kvHeads = configuration.kvHeads
        let encoderLength = configuration.encoderMaxLength
        let headDimensions = configuration.attentionHeadDimensions
        let expectedShape = [
          configuration.decoderLayers,
          1,
          kvHeads,
          encoderLength,
          headDimensions
        ]
        guard projected.shape == expectedShape else {
          throw EdgeToolsCoreMLError(
            code: .invalidStateShape,
            message: "Expected projected encoder shape \(expectedShape), got \(projected.shape)."
          )
        }
        let layerOffset = layer * kvHeads * encoderLength * headDimensions
        let projectedScalars = projected.scalars
        try projectedScalars.withUnsafeBufferPointer { source in
          try state.withMultiArray(for: name) { array in
            let stateShape = array.shape.map(\.intValue)
            let expectedStateShape = [1, kvHeads, encoderLength, headDimensions]
            guard stateShape == expectedStateShape else {
              throw EdgeToolsCoreMLError(
                code: .invalidStateShape,
                message: "Expected state shape \(expectedStateShape), got \(stateShape)."
              )
            }
            array.withUnsafeMutableBufferPointer(ofType: Float16.self) { destination, strides in
              for head in 0..<kvHeads {
                for token in 0..<encoderLength {
                  let sourceOffset =
                    layerOffset + (head * encoderLength + token) * headDimensions
                  let destinationOffset =
                    head * strides[1] + token * strides[2]
                  destination.baseAddress!
                    .advanced(by: destinationOffset)
                    .update(
                      from: source.baseAddress!.advanced(by: sourceOffset),
                      count: headDimensions
                    )
                }
              }
            }
          }
        }
      }
    }
  }

  // MARK: - EdgeToolsCoreMLError

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public struct EdgeToolsCoreMLError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidStateShape = Self(rawValue: "invalid-state-shape")
      public static let missingAOTModel = Self(rawValue: "missing-aot-model")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
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
  extension NeedleCoreMLModel {
    private static func selfAttentionMask(step: Int, maxLength: Int) -> MLTensor {
      let scalars = (0..<maxLength).map {
        $0 <= step ? Float16.zero : Float16(NeedleNumerics.maskedAttentionScore)
      }
      return MLTensor(shape: [1, 1, 1, maxLength], scalars: scalars)
    }
  }

  private enum ModelName {
    static let encoder = "encoder"
    static let decoder = "decoder"
  }

#endif
