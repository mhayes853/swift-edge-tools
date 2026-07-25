#if ONNXCore
  #if Foundation
    import Foundation
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

    struct Sessions: Sendable {
      let encoder: Runtime.Session
      let decoder: Runtime.Session
    }

    private struct EncoderOutputs: Sendable {
      let crossAttentionMask: Runtime.Tensor
      let encoderProjectedK: Runtime.Tensor
      let encoderProjectedV: Runtime.Tensor
    }

    public struct GenerationState: Sendable {
      let crossAttentionMask: Runtime.Tensor
      let encoderProjectedK: Runtime.Tensor
      let encoderProjectedV: Runtime.Tensor
      var keyCache: Runtime.Tensor
      var valueCache: Runtime.Tensor
      var position: Int
    }

    public var vocabularySize: Int { self.configuration.vocabularySize }

    let configuration: NeedleModelConfiguration
    let sessions: Sessions

    public init(
      configuration: NeedleModelConfiguration,
      encoderSession: Runtime.Session,
      decoderSession: Runtime.Session
    ) {
      self.configuration = configuration
      self.sessions = Sessions(encoder: encoderSession, decoder: decoderSession)
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
          name: TensorName.cachePosition
        )
      let cachePosition = try runtime.tensor(values: [position], shape: [1])
      let selfAttentionMask = try runtime.tensor(
        values: Self.selfAttentionMask(
          step: Int(position),
          maxLength: self.configuration.encoderMaxLength
        ),
        shape: [1, 1, 1, self.configuration.encoderMaxLength]
      )
      var outputs = try await self.sessions.decoder.run(
        inputs: [
          TensorName.inputIDs: inputIDs,
          TensorName.cachePosition: cachePosition,
          TensorName.selfAttentionMask: selfAttentionMask,
          TensorName.crossAttentionMask: state.crossAttentionMask,
          TensorName.encoderProjectedK: state.encoderProjectedK,
          TensorName.encoderProjectedV: state.encoderProjectedV,
          TensorName.keyCache: state.keyCache,
          TensorName.valueCache: state.valueCache
        ],
        outputNames: [
          TensorName.logits,
          TensorName.updatedKeyCache,
          TensorName.updatedValueCache
        ]
      )
      guard
        let logits = outputs.removeValue(forKey: TensorName.logits),
        let updatedKey = outputs.removeValue(forKey: TensorName.updatedKeyCache),
        let updatedValue = outputs.removeValue(forKey: TensorName.updatedValueCache)
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      state.keyCache = updatedKey
      state.valueCache = updatedValue
      state.position += 1
      return try await logits.floatValues()
    }

    private func encoderOutputs(
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
      var outputs = try await self.sessions.encoder.run(
        inputs: [TensorName.inputIDs: inputIDs],
        outputNames: [
          TensorName.crossAttentionMask,
          TensorName.encoderProjectedK,
          TensorName.encoderProjectedV
        ]
      )
      guard
        let crossAttentionMask = outputs.removeValue(forKey: TensorName.crossAttentionMask),
        let encoderProjectedK = outputs.removeValue(forKey: TensorName.encoderProjectedK),
        let encoderProjectedV = outputs.removeValue(forKey: TensorName.encoderProjectedV)
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
        crossAttentionMask: encoderOutputs.crossAttentionMask,
        encoderProjectedK: encoderOutputs.encoderProjectedK,
        encoderProjectedV: encoderOutputs.encoderProjectedV,
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

  // MARK: - Tensor Names

  private enum TensorName {
    static let cachePosition = "cache_position"
    static let crossAttentionMask = "cross_attention_mask"
    static let encoderProjectedK = "encoder_projected_k"
    static let encoderProjectedV = "encoder_projected_v"
    static let inputIDs = "input_ids"
    static let keyCache = "key_cache"
    static let logits = "logits"
    static let selfAttentionMask = "self_attention_mask"
    static let updatedKeyCache = "updated_key_cache"
    static let updatedValueCache = "updated_value_cache"
    static let valueCache = "value_cache"
  }

  // MARK: - C ONNX Runtime Loading

  #if canImport(COnnxRuntime)
    public typealias NeedleONNXEngine =
      EdgeToolsONNXEngine<CONNXRuntime, NeedleONNXModel<CONNXRuntime>>
    public typealias EdgeToolsONNXRuntimeError = CONNXRuntimeError

    public struct NeedleONNXRuntimeConfiguration: Hashable, Sendable {
      public var executionProvider: NeedleONNXExecutionProvider

      public init(executionProvider: NeedleONNXExecutionProvider = .cpu) {
        self.executionProvider = executionProvider
      }
    }

    public enum NeedleONNXExecutionProvider: Hashable, Sendable {
      case cpu
      case coreML(computeUnits: NeedleONNXCoreMLComputeUnits)
      case webGPU
    }

    public enum NeedleONNXCoreMLComputeUnits: String, Hashable, Sendable {
      case all = "ALL"
      case cpuAndGPU = "CPUAndGPU"
      case cpuAndNeuralEngine = "CPUAndNeuralEngine"
      case cpuOnly = "CPUOnly"
    }

    #if Foundation
      extension NeedleONNXModel where Runtime == CONNXRuntime {
        init(
          from directoryURL: URL,
          configuration: NeedleModelConfiguration,
          using runtime: CONNXRuntime,
          runtimeConfiguration: NeedleONNXRuntimeConfiguration
        ) throws {
          let sessionConfiguration = CONNXRuntime.SessionConfiguration(
            executionProviders: runtimeConfiguration.executionProvider.runtimeExecutionProviders
          )
          let encoderSession = try runtime.session(
            modelURL: directoryURL.appending(path: "encoder.onnx"),
            configuration: sessionConfiguration
          )
          try encoderSession.validateSignature(
            inputNames: [TensorName.inputIDs],
            outputNames: [
              TensorName.crossAttentionMask,
              TensorName.encoderProjectedK,
              TensorName.encoderProjectedV
            ]
          )
          let decoderSession = try runtime.session(
            modelURL: directoryURL.appending(path: "decoder.onnx"),
            configuration: sessionConfiguration
          )
          try decoderSession.validateSignature(
            inputNames: [
              TensorName.inputIDs,
              TensorName.cachePosition,
              TensorName.selfAttentionMask,
              TensorName.crossAttentionMask,
              TensorName.encoderProjectedK,
              TensorName.encoderProjectedV,
              TensorName.keyCache,
              TensorName.valueCache
            ],
            outputNames: [
              TensorName.logits,
              TensorName.updatedKeyCache,
              TensorName.updatedValueCache
            ]
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
        #if ONNX
          public convenience init(
            from directoryURL: URL,
            runtimeConfiguration: NeedleONNXRuntimeConfiguration = NeedleONNXRuntimeConfiguration()
          ) async throws {
            let runtime = try CONNXRuntime()
            try await self.init(
              from: directoryURL,
              runtime: runtime,
              model: { directoryURL, configuration, runtime in
                try NeedleONNXModel(
                  from: directoryURL,
                  configuration: configuration,
                  using: runtime,
                  runtimeConfiguration: runtimeConfiguration
                )
              }
            )
          }
        #endif

        #if System && ONNX
          public convenience init(
            from directoryPath: FilePath,
            runtimeConfiguration: NeedleONNXRuntimeConfiguration = NeedleONNXRuntimeConfiguration()
          ) async throws {
            try await self.init(
              from: URL(filePath: directoryPath.string, directoryHint: .isDirectory),
              runtimeConfiguration: runtimeConfiguration
            )
          }
        #endif

        public convenience init(
          api: OpaquePointer,
          from directoryURL: URL,
          runtimeConfiguration: NeedleONNXRuntimeConfiguration = NeedleONNXRuntimeConfiguration()
        ) async throws {
          let runtime = try CONNXRuntime(api: api)
          try await self.init(
            from: directoryURL,
            runtime: runtime,
            model: { directoryURL, configuration, runtime in
              try NeedleONNXModel(
                from: directoryURL,
                configuration: configuration,
                using: runtime,
                runtimeConfiguration: runtimeConfiguration
              )
            }
          )
        }
      }
    #endif

    extension NeedleONNXExecutionProvider {
      fileprivate var runtimeExecutionProviders: [CONNXRuntime.ExecutionProvider] {
        switch self {
        case .cpu:
          []
        case .coreML(let computeUnits):
          [.coreML(computeUnits: computeUnits.rawValue)]
        case .webGPU:
          [.webGPU]
        }
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
