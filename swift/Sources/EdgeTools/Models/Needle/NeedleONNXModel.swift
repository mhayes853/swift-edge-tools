#if XGrammar
  import EdgeToolsXGrammar
#endif

#if ONNXCore
  #if Foundation
    import _EdgeToolsFoundation
  #endif

  #if canImport(COnnxRuntime)
    import COnnxRuntime
  #endif

  #if JS && canImport(JavaScriptKit)
    import JavaScriptKit
  #endif

  // MARK: - NeedleONNXGenerateParameters

  public struct NeedleONNXGenerateParameters: NeedleGenerateParameters {
    public static var `default`: Self { Self() }

    public var sampler: any EdgeToolsSampler<ONNXTensorView<Float>>
    public var processor:
      (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], ONNXTensorView<Float>>)?
    public var maxTokens: Int?
    public var toolCallRange: GrammarToolCallRange

    public init(
      sampler: any EdgeToolsSampler<ONNXTensorView<Float>> = ONNXArgmaxSampler(),
      processor: (
        any EdgeToolsLogitsProcessor<
          [EdgeToolsToken.ID], ONNXTensorView<Float>
        >
      )? = nil,
      maxTokens: Int? = 1024,
      toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0)
    ) {
      self.sampler = sampler
      self.processor = processor
      self.maxTokens = maxTokens
      self.toolCallRange = toolCallRange
    }
  }

  // MARK: - NeedleONNXModel

  public struct NeedleONNXModel<Runtime: ONNXRuntime> {
    public typealias ModelConfiguration = NeedleModelConfiguration

    private struct EncoderOutputs {
      let crossAttentionMask: Runtime.Tensor
      let encoderProjectedK: Runtime.Tensor
      let encoderProjectedV: Runtime.Tensor
    }

    private struct Generation {
      let encoderOutputs: EncoderOutputs
      var keyCache: Runtime.Tensor
      var valueCache: Runtime.Tensor
      var activeCacheLength: Int
      var position: Int
      var logitsTensor: Runtime.Tensor?
      var pendingTokenId: EdgeToolsToken.ID?
    }

    public var vocabularySize: Int {
      self.configuration.vocabularySize
    }

    let configuration: NeedleModelConfiguration

    private let runtime: Runtime
    private let encoderSession: Runtime.Session
    private let decoderSession: Runtime.Session
    private var generation: Generation?

    public init(
      configuration: NeedleModelConfiguration,
      runtime: Runtime,
      encoderSession: Runtime.Session,
      decoderSession: Runtime.Session
    ) {
      self.configuration = configuration
      self.runtime = runtime
      self.encoderSession = encoderSession
      self.decoderSession = decoderSession
    }

    private nonisolated(nonsending) mutating func startGeneration(
      tokenIds: [EdgeToolsToken.ID]
    ) async throws {
      guard tokenIds.count <= self.configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: tokenIds.count,
          maximum: self.configuration.encoderMaxLength
        )
      }

      let encoderOutputs = try await self.encoderOutputs(tokenIDs: tokenIds)
      let activeCacheLength = min(128, self.configuration.decoderMaxLength)
      let cacheShape = [
        self.configuration.decoderLayers,
        activeCacheLength,
        self.configuration.kvHeads,
        self.configuration.attentionHeadDimensions
      ]
      let cacheValueCount = cacheShape.reduce(1, *)
      self.generation = Generation(
        encoderOutputs: encoderOutputs,
        keyCache: try self.runtime.tensor(
          values: [Float](repeating: 0, count: cacheValueCount),
          shape: cacheShape
        ),
        valueCache: try self.runtime.tensor(
          values: [Float](repeating: 0, count: cacheValueCount),
          shape: cacheShape
        ),
        activeCacheLength: activeCacheLength,
        position: 0,
        logitsTensor: nil,
        pendingTokenId: nil
      )
      try await self.runDecoder(tokenID: self.configuration.decoderStartTokenId)
    }

    private nonisolated(nonsending) mutating func runDecoder(
      tokenID: EdgeToolsToken.ID
    ) async throws {
      guard var generation = self.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      guard generation.position < self.configuration.decoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: generation.position + 1,
          maximum: self.configuration.decoderMaxLength
        )
      }
      try await self.growCachesIfNeeded(generation: &generation)
      let inputIDs = try self.runtime.tensor(values: [Int64(tokenID)], shape: [1, 1])
      let position = try Int32(exactly: generation.position)
        .unwrapONNXInteger(name: NeedleExportTensorName.cachePosition)
      let cachePosition = try self.runtime.tensor(values: [position], shape: [1])
      let selfAttentionMask = try self.runtime.tensor(
        values: Self.selfAttentionMask(
          step: Int(position),
          maxLength: generation.activeCacheLength
        ),
        shape: [1, 1, 1, generation.activeCacheLength]
      )
      var outputs = try await self.decoderSession.run(
        inputs: [
          NeedleExportTensorName.inputIDs: inputIDs,
          NeedleExportTensorName.cachePosition: cachePosition,
          NeedleExportTensorName.selfAttentionMask: selfAttentionMask,
          NeedleExportTensorName.crossAttentionMask: generation.encoderOutputs.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK: generation.encoderOutputs.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV: generation.encoderOutputs.encoderProjectedV,
          NeedleExportTensorName.keyCache: generation.keyCache,
          NeedleExportTensorName.valueCache: generation.valueCache
        ],
        outputNames: [
          NeedleExportTensorName.logits,
          NeedleExportTensorName.keyCacheDelta,
          NeedleExportTensorName.valueCacheDelta
        ]
      )
      guard
        let logits = outputs.removeValue(forKey: NeedleExportTensorName.logits),
        let keyDelta = outputs.removeValue(forKey: NeedleExportTensorName.keyCacheDelta),
        let valueDelta = outputs.removeValue(forKey: NeedleExportTensorName.valueCacheDelta)
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      try await self.append(
        delta: keyDelta,
        to: generation.keyCache,
        position: generation.position,
        cacheLength: generation.activeCacheLength
      )
      try await self.append(
        delta: valueDelta,
        to: generation.valueCache,
        position: generation.position,
        cacheLength: generation.activeCacheLength
      )
      generation.position += 1
      generation.logitsTensor = logits
      generation.pendingTokenId = nil
      self.generation = generation
    }

  }

  extension NeedleONNXModel {
    private struct CacheCopyRegion: Hashable, Sendable {
      let sourceLength: Int
      let destinationLength: Int
      let sourcePosition: Int
      let destinationPosition: Int
      let positionCount: Int
    }

    private nonisolated(nonsending) func growCachesIfNeeded(
      generation: inout Generation
    ) async throws {
      guard generation.position == generation.activeCacheLength else { return }
      let newLength = min(
        generation.activeCacheLength * 2,
        self.configuration.decoderMaxLength
      )
      guard newLength > generation.activeCacheLength else { return }

      let cacheShape = [
        self.configuration.decoderLayers,
        newLength,
        self.configuration.kvHeads,
        self.configuration.attentionHeadDimensions
      ]
      let cacheValueCount = cacheShape.reduce(1, *)
      let keyCache = try self.runtime.tensor(
        values: [Float](repeating: 0, count: cacheValueCount),
        shape: cacheShape
      )
      let valueCache = try self.runtime.tensor(
        values: [Float](repeating: 0, count: cacheValueCount),
        shape: cacheShape
      )
      let copyRegion = CacheCopyRegion(
        sourceLength: generation.activeCacheLength,
        destinationLength: newLength,
        sourcePosition: 0,
        destinationPosition: 0,
        positionCount: generation.activeCacheLength
      )
      try await self.copyCacheLayers(
        from: generation.keyCache,
        to: keyCache,
        region: copyRegion
      )
      try await self.copyCacheLayers(
        from: generation.valueCache,
        to: valueCache,
        region: copyRegion
      )
      generation.keyCache = keyCache
      generation.valueCache = valueCache
      generation.activeCacheLength = newLength
    }

    private nonisolated(nonsending) func append(
      delta: Runtime.Tensor,
      to cache: Runtime.Tensor,
      position: Int,
      cacheLength: Int
    ) async throws {
      try await self.copyCacheLayers(
        from: delta,
        to: cache,
        region: CacheCopyRegion(
          sourceLength: 1,
          destinationLength: cacheLength,
          sourcePosition: 0,
          destinationPosition: position,
          positionCount: 1
        )
      )
    }

    private nonisolated(nonsending) func copyCacheLayers(
      from source: Runtime.Tensor,
      to destination: Runtime.Tensor,
      region: CacheCopyRegion
    ) async throws {
      let layers = self.configuration.decoderLayers
      let kvHeads = self.configuration.kvHeads
      let headDimensions = self.configuration.attentionHeadDimensions
      #if JS && canImport(JavaScriptKit)
        if
          let source = source as? JSONNXTensor,
          let destination = destination as? JSONNXTensor
        {
          try await Self.copyJSONCacheLayers(
            from: source,
            to: destination,
            region: region,
            layers: layers,
            rowCount: kvHeads * headDimensions
          )
          return
        }
      #endif

      let rowCount = kvHeads * headDimensions
      try await destination.withMutableView(as: Float.self) { destinationView in
        try await source.withView(as: Float.self) { sourceView in
          var destinationSpan = destinationView.mutableSpan
          let sourceSpan = sourceView.span
          let valueCount = region.positionCount * rowCount
          for layer in 0..<layers {
            let sourceOffset =
              (layer * region.sourceLength + region.sourcePosition) * rowCount
            let destinationOffset =
              (layer * region.destinationLength + region.destinationPosition) * rowCount
            for valueOffset in 0..<valueCount {
              destinationSpan[destinationOffset + valueOffset] =
                sourceSpan[sourceOffset + valueOffset]
            }
          }
        }
      }
    }

    #if JS && canImport(JavaScriptKit)
      private static nonisolated(nonsending) func copyJSONCacheLayers(
        from source: JSONNXTensor,
        to destination: JSONNXTensor,
        region: CacheCopyRegion,
        layers: Int,
        rowCount: Int
      ) async throws {
        let sourceData = try await Self.jsonData(for: source)
        let destinationData = try await Self.jsonData(for: destination)
        let subarray = sourceData["subarray"].object!
        let set = destinationData["set"].object!

        for layer in 0..<layers {
          let sourceOffset =
            (layer * region.sourceLength + region.sourcePosition) * rowCount
          let destinationOffset =
            (layer * region.destinationLength + region.destinationPosition) * rowCount
          let sourceSlice = try subarray.throws(
            this: sourceData,
            sourceOffset,
            sourceOffset + region.positionCount * rowCount
          ).object!
          _ = try set.throws(this: destinationData, sourceSlice, destinationOffset)
        }
      }

      private static nonisolated(nonsending) func jsonData(
        for tensor: JSONNXTensor
      ) async throws -> JSObject {
        let getData = tensor.object["getData"].object!
        return try await JSONNXRuntime.promiseObject(try getData.throws(this: tensor.object))
      }
    #endif

    private nonisolated(nonsending) func encoderOutputs(
      tokenIDs: [EdgeToolsToken.ID]
    ) async throws -> EncoderOutputs {
      let runtime = self.runtime
      var tokens = tokenIDs.map(Int64.init)
      let inputIDs = try runtime.tensor(values: tokens, shape: [1, tokens.count])
      var outputs = try await self.encoderSession.run(
        inputs: [NeedleExportTensorName.inputIDs: inputIDs],
        outputNames: [
          NeedleExportTensorName.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV
        ]
      )
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

    private static func selfAttentionMask(step: Int, maxLength: Int) -> [Float] {
      (0..<maxLength).map { $0 <= step ? 0 : NeedleNumerics.maskedAttentionScore }
    }
  }

  // MARK: - NeedleModel

  extension NeedleONNXModel: NeedleModel {
    public typealias Input = [EdgeToolsToken.ID]
    public typealias GenerateParameters = NeedleONNXGenerateParameters

    public func input(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> EdgeToolsModelInput<[EdgeToolsToken.ID]> {
      let tokenIds = try tokenizer.encode(text: prompt.formatted(tools: tools))
      return EdgeToolsModelInput(value: tokenIds, tokenIds: tokenIds)
    }

    public nonisolated(nonsending) mutating func prepare(
      input: [EdgeToolsToken.ID],
      parameters: NeedleONNXGenerateParameters
    ) async throws -> EdgeToolsModelPreparation {
      let clock = ContinuousClock()
      let start = clock.now
      parameters.processor?.prompt(input)
      try await self.startGeneration(tokenIds: input)
      return EdgeToolsModelPreparation(
        metrics: EdgeToolsPrefillMetrics(
          tokens: input.count,
          duration: start.duration(to: clock.now)
        )
      )
    }

    public nonisolated(nonsending) mutating func decode(
      bitmask: GrammarBitmask,
      parameters: NeedleONNXGenerateParameters
    ) async throws -> EdgeToolsModelSample {
      if let pendingTokenId = self.generation?.pendingTokenId {
        try await self.runDecoder(tokenID: pendingTokenId)
      }
      guard var generation = self.generation, let logitsTensor = generation.logitsTensor else {
        throw EdgeToolsError.modelNotPrepared
      }
      let vocabularySize = self.vocabularySize
      let (tokenId, confidence) = try await logitsTensor.withMutableView(
        as: Float.self
      ) { logits -> (EdgeToolsToken.ID, Float) in
        guard logits.count == vocabularySize else {
          throw ONNXError(
            code: .invalidLogitsCount,
            message: "Expected \(vocabularySize) logits, got \(logits.count)."
          )
        }
        try await parameters.processor?.process(logits: &logits)
        var span = logits.mutableSpan
        applyBitmask(logits: &span, mask: bitmask)
        let confidence = tokenConfidenceONNX(logits: logits.span)
        let tokenId = try await parameters.sampler.sample(logits: logits)
        return (tokenId, confidence)
      }
      parameters.processor?.didSample(tokenId: tokenId)
      generation.pendingTokenId = tokenId
      self.generation = generation
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public mutating func resetGeneration() {
      self.generation = nil
    }
  }

  public typealias NeedleONNXModelEngine<Runtime: ONNXRuntime> =
    EdgeToolsModelEngine<NeedleONNXModel<Runtime>>

  // MARK: - C ONNX Runtime Loading

  #if canImport(COnnxRuntime)
    public typealias NeedleCONNXModelEngine =
      EdgeToolsModelEngine<NeedleONNXModel<CONNXRuntime>>

    #if Foundation
      extension EdgeToolsModelEngine where Model == NeedleONNXModel<CONNXRuntime> {
        public init(
          from directoryURL: URL,
          runtime: sending CONNXRuntime
        ) async throws {
          let tokenizer = try await loadEdgeToolsTokenizer(from: directoryURL)
          guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
            throw EdgeToolsError.unsupportedTokenizer
          }
          guard let configuration = try NeedleModelConfiguration.decode(in: directoryURL) else {
            throw EdgeToolsError.failedToLoadConfiguration
          }
          let encoderSession = try runtime.session(
            modelPath: directoryURL.appending(path: "encoder.onnx").path()
          )
          let decoderSession = try runtime.session(
            modelPath: directoryURL.appending(path: "decoder.onnx").path(),
            configuration: CONNXRuntime.Configuration(
              configureSessionOptions: { runtime, handle in
                try runtime.check(runtime.api.pointee.SetIntraOpNumThreads(handle, 1))
              }
            )
          )
          let model = NeedleONNXModel(
            configuration: configuration,
            runtime: runtime,
            encoderSession: encoderSession,
            decoderSession: decoderSession
          )
          try self.init(model: model, tokenizer: tokenizer)
        }

        #if ONNX
          public init(
            from directoryURL: URL,
            runtimeConfiguration: CONNXRuntime.Configuration = CONNXRuntime.Configuration()
          ) async throws {
            let runtime = try CONNXRuntime(configuration: runtimeConfiguration)
            try await self.init(from: directoryURL, runtime: runtime)
          }
        #endif

        public init(
          api: OpaquePointer,
          from directoryURL: URL,
          runtimeConfiguration: CONNXRuntime.Configuration = CONNXRuntime.Configuration()
        ) async throws {
          let runtime = try CONNXRuntime(api: api, configuration: runtimeConfiguration)
          try await self.init(from: directoryURL, runtime: runtime)
        }
      }
    #endif
  #endif

  // MARK: - JavaScript ONNX Runtime Loading

  #if JS && canImport(JavaScriptKit)
    public typealias NeedleJSONNXModelEngine =
      EdgeToolsModelEngine<NeedleONNXModel<JSONNXRuntime>>

    extension EdgeToolsModelEngine where Model == NeedleONNXModel<JSONNXRuntime> {
      public init(
        onnxRuntime: sending JSObject,
        configuration: NeedleModelConfiguration,
        tokenizer: sending NeedleSPTokenizer,
        encoderModel: JSONNXRuntime.ModelSource,
        decoderModel: JSONNXRuntime.ModelSource,
        encoderConfiguration: JSONNXRuntime.Configuration = JSONNXRuntime.Configuration(),
        decoderConfiguration: JSONNXRuntime.Configuration = JSONNXRuntime.Configuration()
      ) async throws {
        configureNeedleBrowserWasmThreads(onnxRuntime: onnxRuntime)
        let runtime = try JSONNXRuntime(onnxRuntime: onnxRuntime)
        let encoderSession = try await runtime.session(
          model: encoderModel,
          configuration: encoderConfiguration
        )
        let decoderSession = try await runtime.session(
          model: decoderModel,
          configuration: decoderConfiguration
        )
        let model = NeedleONNXModel<JSONNXRuntime>(
          configuration: configuration,
          runtime: consume runtime,
          encoderSession: consume encoderSession,
          decoderSession: consume decoderSession
        )
        try self.init(model: consume model, tokenizer: tokenizer)
      }
    }

    private func configureNeedleBrowserWasmThreads(onnxRuntime: JSObject) {
      let object = onnxRuntime["default"].object ?? onnxRuntime
      guard
        let environment = object["env"].object,
        environment["versions"].object?["web"].string != nil,
        let wasm = environment["wasm"].object
      else { return }
      wasm["numThreads"] = .number(4)
    }
  #endif

  // MARK: - Helpers

  extension Optional where Wrapped: FixedWidthInteger {
    fileprivate func unwrapONNXInteger(name: String) throws -> Wrapped {
      guard let value = self else {
        throw ONNXError(
          code: .integerConversionFailure,
          message: "Value for \(name) cannot be represented by the ONNX model's integer type."
        )
      }
      return value
    }
  }
#endif
