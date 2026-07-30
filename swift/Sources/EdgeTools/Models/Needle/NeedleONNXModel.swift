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

  // MARK: - NeedleONNXModelAssets

  public struct NeedleONNXModelAssets<Runtime: EdgeToolsONNXRuntime> {
    let runtime: Runtime
    let encoderSession: Runtime.Session
    let decoderSession: Runtime.Session
  }

  // MARK: - NeedleONNXModel

  public struct NeedleONNXModel<Runtime: EdgeToolsONNXRuntime> {
    public typealias ModelConfiguration = NeedleModelConfiguration
    public typealias Prompt = NeedlePrompt
    public typealias ToolCallParser = NeedleToolCallParser

    fileprivate struct EncoderOutputs {
      let crossAttentionMask: Runtime.Tensor
      let encoderProjectedK: Runtime.Tensor
      let encoderProjectedV: Runtime.Tensor
    }

    public struct ModelState {
      fileprivate let encoderOutputs: EncoderOutputs
      var keyCache: Runtime.Tensor
      var valueCache: Runtime.Tensor
      var position: Int
    }

    public var vocabularySize: Int { self.configuration.vocabularySize }

    let configuration: NeedleModelConfiguration

    public init(configuration: NeedleModelConfiguration) {
      self.configuration = configuration
    }

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.needle(tools: tools, range: range)
    }

    private nonisolated(nonsending) func prepareModel(
      tokenIds: [EdgeToolsToken.ID],
      assets: NeedleONNXModelAssets<Runtime>
    ) async throws -> (logits: [Float], state: ModelState) {
      guard tokenIds.count <= self.configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: tokenIds.count,
          maximum: self.configuration.encoderMaxLength
        )
      }

      let encoderOutputs = try await self.encoderOutputs(
        tokenIDs: tokenIds,
        assets: assets
      )
      var state = try self.initialGenerationState(
        encoderOutputs: encoderOutputs,
        using: assets.runtime
      )
      let logits = try await self.decode(
        tokenID: self.configuration.decoderStartTokenId,
        state: &state,
        assets: assets
      )
      return (logits, state)
    }

    private nonisolated(nonsending) func decode(
      tokenID: EdgeToolsToken.ID,
      state: inout ModelState,
      assets: NeedleONNXModelAssets<Runtime>
    ) async throws -> [Float] {
      let runtime = assets.runtime
      let inputIDs = try runtime.tensor(values: [Int64(tokenID)], shape: [1, 1])
      let position = try Int32(exactly: state.position)
        .unwrapONNXInteger(
          name: NeedleExportTensorName.cachePosition
        )
      let cachePosition = try runtime.tensor(values: [position], shape: [1])
      let selfAttentionMask = try runtime.tensor(
        values: Self.selfAttentionMask(
          step: Int(position),
          maxLength: self.configuration.encoderMaxLength
        ),
        shape: [1, 1, 1, self.configuration.encoderMaxLength]
      )
      var outputs = try await assets.decoderSession.run(
        inputs: [
          NeedleExportTensorName.inputIDs: inputIDs,
          NeedleExportTensorName.cachePosition: cachePosition,
          NeedleExportTensorName.selfAttentionMask: selfAttentionMask,
          NeedleExportTensorName.crossAttentionMask: state.encoderOutputs.crossAttentionMask,
          NeedleExportTensorName.encoderProjectedK: state.encoderOutputs.encoderProjectedK,
          NeedleExportTensorName.encoderProjectedV: state.encoderOutputs.encoderProjectedV,
          NeedleExportTensorName.keyCache: state.keyCache,
          NeedleExportTensorName.valueCache: state.valueCache
        ],
        outputNames: [
          NeedleExportTensorName.logits,
          NeedleExportTensorName.updatedKeyCache,
          NeedleExportTensorName.updatedValueCache
        ]
      )
      guard
        let logits = outputs.removeValue(forKey: NeedleExportTensorName.logits),
        let updatedKey = outputs.removeValue(forKey: NeedleExportTensorName.updatedKeyCache),
        let updatedValue = outputs.removeValue(forKey: NeedleExportTensorName.updatedValueCache)
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      state.keyCache = updatedKey
      state.valueCache = updatedValue
      state.position += 1
      return try await logits.array(as: Float.self)
    }

    private nonisolated(nonsending) func encoderOutputs(
      tokenIDs: [EdgeToolsToken.ID],
      assets: NeedleONNXModelAssets<Runtime>
    ) async throws -> EncoderOutputs {
      let runtime = assets.runtime
      var paddedTokens = tokenIDs.map(Int64.init)
      paddedTokens.append(
        contentsOf: repeatElement(
          Int64(self.configuration.padTokenId),
          count: self.configuration.encoderMaxLength - paddedTokens.count
        )
      )
      let inputIDs = try runtime.tensor(
        values: paddedTokens,
        shape: [1, self.configuration.encoderMaxLength]
      )
      var outputs = try await assets.encoderSession.run(
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

    private func initialGenerationState(
      encoderOutputs: EncoderOutputs,
      using runtime: Runtime
    ) throws -> ModelState {
      let cacheShape = [
        self.configuration.decoderLayers,
        self.configuration.encoderMaxLength,
        self.configuration.attentionHeads,
        self.configuration.attentionHeadDimensions
      ]
      let cacheValueCount = cacheShape.reduce(1, *)
      return ModelState(
        encoderOutputs: encoderOutputs,
        keyCache: try runtime.tensor(
          values: [Float](repeating: 0, count: cacheValueCount),
          shape: cacheShape
        ),
        valueCache: try runtime.tensor(
          values: [Float](repeating: 0, count: cacheValueCount),
          shape: cacheShape
        ),
        position: 0
      )
    }

    private static func selfAttentionMask(step: Int, maxLength: Int) -> [Float] {
      let allowedStart = max(0, maxLength - step - 1)
      return (0..<maxLength).map { $0 < allowedStart ? -65500 : 0 }
    }
  }

  // MARK: - EdgeToolsModel

  extension NeedleONNXModel: EdgeToolsModel {
    public typealias Input = [EdgeToolsToken.ID]
    public typealias Logits = [Float]
    public typealias Assets = NeedleONNXModelAssets<Runtime>
    public typealias GenerateParameters = EdgeToolsONNXGenerateParameters
    public typealias GenerationState = EdgeToolsONNXGenerationState<NeedleONNXModel.ModelState>

    public func input(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> [EdgeToolsToken.ID] {
      try tokenizer.encode(text: prompt.formatted(tools: tools))
    }

    public func tokenIds(
      in input: [EdgeToolsToken.ID]
    ) -> [EdgeToolsToken.ID] {
      input
    }

    public func prepare(
      input: [EdgeToolsToken.ID],
      parameters: EdgeToolsONNXGenerateParameters,
      assets: NeedleONNXModelAssets<Runtime>
    ) async throws -> EdgeToolsModelPreparation<[Float], GenerationState> {
      let clock = ContinuousClock()
      let start = clock.now
      let processor = parameters.processor
      processor?.prompt(input)
      let preparation = try await self.prepareModel(tokenIds: input, assets: assets)
      let state = EdgeToolsONNXGenerationState(
        modelState: preparation.state,
        sampler: parameters.sampler,
        processor: processor
      )
      return EdgeToolsModelPreparation(
        logits: preparation.logits,
        state: state,
        metrics: EdgeToolsPrefillMetrics(
          tokens: input.count,
          duration: start.duration(to: clock.now)
        )
      )
    }

    public func decode(
      tokenId: EdgeToolsToken.ID,
      state: inout GenerationState,
      assets: NeedleONNXModelAssets<Runtime>
    ) async throws -> [Float] {
      try await self.decode(
        tokenID: tokenId,
        state: &state.modelState,
        assets: assets
      )
    }

    public func sample(
      logits: inout [Float],
      bitmask: GrammarBitmask,
      state: inout GenerationState
    ) async throws -> EdgeToolsModelSample {
      guard logits.count == self.vocabularySize else {
        throw EdgeToolsONNXError(
          code: .invalidLogitsCount,
          message: "Expected \(self.vocabularySize) logits, got \(logits.count)."
        )
      }
      var logitsView = EdgeToolsONNXTensorView(copying: logits)
      try await state.processor?.process(logits: &logitsView)
      applyONNXBitmask(logits: &logitsView, mask: bitmask)
      let confidence = tokenConfidenceONNX(logits: logitsView)
      let tokenId = try await state.sampler.sample(logits: logitsView)
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public func didAccept(token: EdgeToolsToken, state: inout GenerationState) {
      state.processor?.didSample(token: token)
    }
  }

  public typealias NeedleONNXModelEngine<Runtime: EdgeToolsONNXRuntime> =
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
            modelURL: directoryURL.appending(path: "encoder.onnx")
          )
          let decoderSession = try runtime.session(
            modelURL: directoryURL.appending(path: "decoder.onnx")
          )
          let assets = NeedleONNXModelAssets(
            runtime: runtime,
            encoderSession: encoderSession,
            decoderSession: decoderSession
          )
          try self.init(
            model: NeedleONNXModel(configuration: configuration),
            assets: assets,
            tokenizer: tokenizer
          )
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
        let runtime = try JSONNXRuntime(onnxRuntime: onnxRuntime)
        let encoderSession = try await runtime.session(
          model: encoderModel,
          configuration: encoderConfiguration
        )
        let decoderSession = try await runtime.session(
          model: decoderModel,
          configuration: decoderConfiguration
        )
        let assets = NeedleONNXModelAssets(
          runtime: consume runtime,
          encoderSession: consume encoderSession,
          decoderSession: consume decoderSession
        )
        try self.init(
          model: NeedleONNXModel<JSONNXRuntime>(configuration: configuration),
          assets: consume assets,
          tokenizer: tokenizer
        )
      }
    }
  #endif

  // MARK: - Helpers

  extension Optional where Wrapped: FixedWidthInteger {
    fileprivate func unwrapONNXInteger(name: String) throws -> Wrapped {
      guard let value = self else {
        throw EdgeToolsONNXError(
          code: .integerConversionFailure,
          message: "Value for \(name) cannot be represented by the ONNX model's integer type."
        )
      }
      return value
    }
  }
#endif
