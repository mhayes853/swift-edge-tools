#if XGrammar
  import EdgeToolsXGrammar
#endif

#if ONNXCore
  #if Foundation
    public import _EdgeToolsFoundation
  #endif

  #if Foundation && System
    import SystemPackage
  #endif

  #if canImport(COnnxRuntime)
    import COnnxRuntime
  #endif

  // MARK: - NeedleONNXModel

  public struct NeedleONNXModel<Runtime: EdgeToolsONNXRuntime>: EdgeToolsONNXModel {
    public typealias ModelConfiguration = NeedleModelConfiguration
    public typealias Prompt = NeedlePrompt
    public typealias ToolCallParser = NeedleToolCallParser

    fileprivate struct EncoderOutputs {
      let crossAttentionMask: Runtime.Tensor
      let encoderProjectedK: Runtime.Tensor
      let encoderProjectedV: Runtime.Tensor
    }

    public struct GenerationState {
      fileprivate let encoderOutputs: EncoderOutputs
      var keyCache: Runtime.Tensor
      var valueCache: Runtime.Tensor
      var position: Int
    }

    public var vocabularySize: Int { self.configuration.vocabularySize }

    let configuration: NeedleModelConfiguration
    let encoderSession: Runtime.Session
    let decoderSession: Runtime.Session

    public init(
      configuration: NeedleModelConfiguration,
      encoderSession: Runtime.Session,
      decoderSession: Runtime.Session
    ) {
      self.configuration = configuration
      self.encoderSession = encoderSession
      self.decoderSession = decoderSession
    }

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.needle(tools: tools, range: range)
    }

    public func process(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> [EdgeToolsToken.ID] {
      try tokenizer.encode(text: prompt.formatted(tools: tools))
    }

    public func prepare(
      tokenIDs: [EdgeToolsToken.ID],
      using runtime: Runtime
    ) async throws -> EdgeToolsONNXModelPreparation<GenerationState> {
      guard tokenIDs.count <= self.configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: tokenIDs.count,
          maximum: self.configuration.encoderMaxLength
        )
      }

      let encoderOutputs = try await self.encoderOutputs(
        tokenIDs: tokenIDs,
        using: runtime
      )
      var state = try self.initialGenerationState(
        encoderOutputs: encoderOutputs,
        using: runtime
      )
      let logits = try await self.decode(
        tokenID: self.configuration.decoderStartTokenId,
        state: &state,
        using: runtime
      )
      return EdgeToolsONNXModelPreparation(logits: logits, state: state)
    }

    public func decode(
      tokenID: EdgeToolsToken.ID,
      state: inout GenerationState,
      using runtime: Runtime
    ) async throws -> [Float] {
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
      var outputs = try await self.decoderSession.run(
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
      return try await logits.floatValues()
    }

    private nonisolated(nonsending) func encoderOutputs(
      tokenIDs: [EdgeToolsToken.ID],
      using runtime: Runtime
    ) async throws -> EncoderOutputs {
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

    private func initialGenerationState(
      encoderOutputs: EncoderOutputs,
      using runtime: Runtime
    ) throws -> GenerationState {
      let cacheShape = [
        self.configuration.decoderLayers,
        self.configuration.encoderMaxLength,
        self.configuration.attentionHeads,
        self.configuration.attentionHeadDimensions
      ]
      let cacheValueCount = cacheShape.reduce(1, *)
      return GenerationState(
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

  // MARK: - C ONNX Runtime Loading

  #if canImport(COnnxRuntime)
    public typealias NeedleONNXEngine =
      EdgeToolsONNXEngine<CONNXRuntime, NeedleONNXModel<CONNXRuntime>>
    public typealias EdgeToolsONNXRuntimeError = CONNXRuntimeError

    #if Foundation
      extension NeedleONNXModel where Runtime == CONNXRuntime {
        init(
          from directoryURL: URL,
          configuration: NeedleModelConfiguration,
          using runtime: CONNXRuntime
        ) throws {
          let encoderSession = try runtime.session(
            modelURL: directoryURL.appending(path: "encoder.onnx")
          )
          let decoderSession = try runtime.session(
            modelURL: directoryURL.appending(path: "decoder.onnx")
          )
          self.init(
            configuration: configuration,
            encoderSession: encoderSession,
            decoderSession: decoderSession
          )
        }
      }

      extension EdgeToolsONNXEngine
      where Runtime == CONNXRuntime, Model == NeedleONNXModel<CONNXRuntime> {
        public init(
          from directoryURL: URL,
          runtime: sending CONNXRuntime
        ) async throws {
          try await self.init(
            from: directoryURL,
            runtime: runtime,
            model: { directoryURL, configuration, runtime in
              try NeedleONNXModel(
                from: directoryURL,
                configuration: configuration,
                using: runtime
              )
            }
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

        #if System && ONNX
          public init(
            from directoryPath: FilePath,
            runtimeConfiguration: CONNXRuntime.Configuration = CONNXRuntime.Configuration()
          ) async throws {
            try await self.init(
              from: URL(filePath: directoryPath.string, directoryHint: .isDirectory),
              runtimeConfiguration: runtimeConfiguration
            )
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
