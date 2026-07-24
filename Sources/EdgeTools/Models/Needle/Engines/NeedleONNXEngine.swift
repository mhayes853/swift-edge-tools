#if ONNXCore
  import Atomics
  import Foundation

  #if canImport(COnnxRuntime)
    import COnnxRuntime
  #endif

  public protocol NeedleONNXBackend: Sendable {
    var configuration: NeedleModelConfiguration { get }

    func prefill(
      tokenIDs: [EdgeToolsToken.ID]
    ) async throws -> any NeedleONNXGeneration
  }

  public protocol NeedleONNXGeneration: Sendable {
    func decode(
      tokenID: EdgeToolsToken.ID,
      position: Int
    ) async throws -> [Float]
  }

  public final class NeedleONNXEngine: EdgeToolsEngine {
    public typealias Prompt = NeedlePrompt

    public struct RuntimeConfiguration: Hashable, Sendable {
      public var executionProvider: ExecutionProvider

      public init(executionProvider: ExecutionProvider = .cpu) {
        self.executionProvider = executionProvider
      }
    }

    public enum ExecutionProvider: Hashable, Sendable {
      case cpu
      case coreML(computeUnits: CoreMLComputeUnits)
      case webGPU
    }

    public enum CoreMLComputeUnits: String, Hashable, Sendable {
      case all = "ALL"
      case cpuAndGPU = "CPUAndGPU"
      case cpuAndNeuralEngine = "CPUAndNeuralEngine"
      case cpuOnly = "CPUOnly"
    }

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self { Self() }

      private var _sampler: @Sendable () -> any EdgeToolsSampler<[Float]>
      private var _processor:
        @Sendable () -> (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)?

      public var sampler: any EdgeToolsSampler<[Float]> {
        self._sampler()
      }

      public var processor: (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)? {
        self._processor()
      }

      public var toolCallRange: GrammarToolCallRange
      public var maxTokens: Int?

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any EdgeToolsSampler<[Float]> =
          ONNXArgmaxSampler(),
        processor:
          @autoclosure @escaping @Sendable () ->
          (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)? = nil,
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
      let grammarEngine: XGRCompiler
      let matcherPool: XGRToolCallMatcherPool
    }

    private let state: Lock<State>
    private let backend: any NeedleONNXBackend
    private let configuration: NeedleModelConfiguration
    private let tokenizer: any EdgeToolsXGRTokenizer
    private let clock = ContinuousClock()

    #if ONNX && canImport(COnnxRuntime)
      public convenience init(
        modelDirectoryURL: URL,
        runtimeConfiguration: RuntimeConfiguration = RuntimeConfiguration(),
        editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
      ) async throws {
        let tokenizer = try await loadEdgeToolsTokenizer(from: modelDirectoryURL)
        guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }

        var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
        editConfiguration(&configuration)
        let backend = try NeedleONNXRuntimeBackend(
          modelDirectoryURL: modelDirectoryURL,
          runtimeConfiguration: runtimeConfiguration,
          configuration: configuration
        )
        try self.init(backend: backend, tokenizer: tokenizer)
      }
    #endif

    public init(
      backend: sending any NeedleONNXBackend,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      let configuration = backend.configuration
      let grammarEngine = try XGRCompiler(
        tokenizerInfo: try tokenizer.tokenizerInfo(
          modelVocabularySize: configuration.vocabularySize
        )
      )
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          matcherPool: XGRToolCallMatcherPool.needle()
        )
      )
      self.backend = backend
      self.configuration = configuration
      self.tokenizer = tokenizer
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
      isStopped: ManagedAtomic<Bool>
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = consume matcher
      let sampler = parameters.sampler
      var processor = parameters.processor
      let generateStart = self.clock.now
      let (generation, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        tools: tools,
        processor: &processor
      )

      var nextDecoderTokenID = self.configuration.decoderStartTokenId
      var loop = EdgeToolsGenerationLoop<NeedleToolCallParser>(
        matcher: consume matcher,
        tokenizer: self.tokenizer,
        channel: channel,
        isStopped: isStopped,
        maximumTokenCount: parameters.maxTokens,
        generateStart: generateStart
      )
      while let bitmask = try loop.nextBitmask() {
        var logits = try await generation.decode(
          tokenID: nextDecoderTokenID,
          position: loop.generatedTokenCount
        )
        guard logits.count == self.configuration.vocabularySize else {
          throw EdgeToolsONNXError(
            code: .invalidLogitsCount,
            message: "Expected \(self.configuration.vocabularySize) logits, got \(logits.count)."
          )
        }
        logits = try await processor?.process(logits: &logits) ?? logits
        applyONNXBitmask(logits: &logits, mask: bitmask)
        let confidence = tokenConfidenceONNX(logits: logits)
        let tokenID = try await sampler.sample(logits: logits)
        let token = try loop.accept(tokenID: tokenID, confidence: confidence)
        nextDecoderTokenID = tokenID
        processor?.didSample(token: token)
      }

      return loop.finish(prefillMetrics: prefillMetrics)
    }

    private func prefill(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      processor: inout (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)?
    ) async throws -> (any NeedleONNXGeneration, EdgeToolsPrefillMetrics) {
      let promptTokens = try self.tokenizer.encode(text: prompt.formatted(tools: tools))
      guard promptTokens.count <= self.configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: self.configuration.encoderMaxLength
        )
      }

      let prefillStart = self.clock.now
      processor?.prompt(promptTokens)
      let generation = try await self.backend.prefill(tokenIDs: promptTokens)
      return (
        generation,
        EdgeToolsPrefillMetrics(
          tokens: promptTokens.count,
          duration: prefillStart.duration(to: self.clock.now)
        )
      )
    }

    private static func decodeConfiguration(from directory: URL) throws -> NeedleModelConfiguration
    {
      guard let configuration = try NeedleModelConfiguration.decode(in: directory) else {
        throw EdgeToolsError.failedToLoadConfiguration
      }
      return configuration
    }
  }

  public struct EdgeToolsONNXError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidLogitsCount = Self(rawValue: "invalid-logits-count")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  #if canImport(COnnxRuntime)
    public final class NeedleCONNXRuntimeBackend: NeedleONNXBackend {
      public let configuration: NeedleModelConfiguration

      private let runtime: CONNXRuntime
      private let encoderSession: CONNXRuntimeSession
      private let decoderSession: CONNXRuntimeSession

      public convenience init(
        api: OpaquePointer,
        modelDirectoryURL: URL,
        runtimeConfiguration: NeedleONNXEngine.RuntimeConfiguration =
          NeedleONNXEngine.RuntimeConfiguration(),
        configuration: NeedleModelConfiguration
      ) throws {
        try self.init(
          runtime: CONNXRuntime(api: api),
          modelDirectoryURL: modelDirectoryURL,
          runtimeConfiguration: runtimeConfiguration,
          configuration: configuration
        )
      }

      #if ONNX
        public convenience init(
          modelDirectoryURL: URL,
          runtimeConfiguration: NeedleONNXEngine.RuntimeConfiguration =
            NeedleONNXEngine.RuntimeConfiguration(),
          configuration: NeedleModelConfiguration
        ) throws {
          try self.init(
            runtime: CONNXRuntime(),
            modelDirectoryURL: modelDirectoryURL,
            runtimeConfiguration: runtimeConfiguration,
            configuration: configuration
          )
        }
      #endif

      public init(
        runtime: CONNXRuntime,
        modelDirectoryURL: URL,
        runtimeConfiguration: NeedleONNXEngine.RuntimeConfiguration =
          NeedleONNXEngine.RuntimeConfiguration(),
        configuration: NeedleModelConfiguration
      ) throws {
        let sessionConfiguration = CONNXRuntime.SessionConfiguration(
          executionProviders: runtimeConfiguration.executionProvider.runtimeExecutionProviders
        )
        let encoderSession = try runtime.session(
          modelURL: modelDirectoryURL.appending(path: "encoder.onnx"),
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
          modelURL: modelDirectoryURL.appending(path: "decoder.onnx"),
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
        self.configuration = configuration
        self.runtime = runtime
        self.encoderSession = encoderSession
        self.decoderSession = decoderSession
      }

      public func prefill(
        tokenIDs: [EdgeToolsToken.ID]
      ) async throws -> any NeedleONNXGeneration {
        var paddedTokens = tokenIDs.map(Int64.init)
        paddedTokens.append(
          contentsOf: repeatElement(
            Int64(self.configuration.padTokenId),
            count: self.configuration.encoderMaxLength - paddedTokens.count
          )
        )
        let inputIDs = try self.runtime.tensor(
          values: paddedTokens,
          shape: [1, self.configuration.encoderMaxLength]
        )
        var outputs = try self.encoderSession.run(
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
        return try NeedleCONNXRuntimeGeneration(
          runtime: self.runtime,
          decoderSession: self.decoderSession,
          configuration: self.configuration,
          encoderOutputs: NeedleCONNXRuntimeGeneration.EncoderOutputs(
            crossAttentionMask: crossAttentionMask,
            encoderProjectedK: encoderProjectedK,
            encoderProjectedV: encoderProjectedV
          )
        )
      }
    }

    public typealias NeedleONNXRuntimeBackend = NeedleCONNXRuntimeBackend
    public typealias EdgeToolsONNXRuntimeError = CONNXRuntimeError

    private final class NeedleCONNXRuntimeGeneration: NeedleONNXGeneration {
      struct EncoderOutputs: Sendable {
        let crossAttentionMask: CONNXRuntimeTensor
        let encoderProjectedK: CONNXRuntimeTensor
        let encoderProjectedV: CONNXRuntimeTensor
      }

      private struct Cache: Sendable {
        var key: CONNXRuntimeTensor
        var value: CONNXRuntimeTensor
      }

      private struct State: ~Copyable {
        var cache: Cache
      }

      private let runtime: CONNXRuntime
      private let decoderSession: CONNXRuntimeSession
      private let configuration: NeedleModelConfiguration
      private let encoderOutputs: EncoderOutputs
      private let state: Lock<State>

      init(
        runtime: CONNXRuntime,
        decoderSession: CONNXRuntimeSession,
        configuration: NeedleModelConfiguration,
        encoderOutputs: EncoderOutputs
      ) throws {
        let cacheShape = [
          configuration.decoderLayers,
          configuration.encoderMaxLength,
          configuration.attentionHeads,
          configuration.attentionHeadDimensions
        ]
        self.runtime = runtime
        self.decoderSession = decoderSession
        self.configuration = configuration
        self.encoderOutputs = encoderOutputs
        self.state = Lock(
          State(
            cache: Cache(
              key: try runtime.tensor(repeating: 0, shape: cacheShape),
              value: try runtime.tensor(repeating: 0, shape: cacheShape)
            )
          )
        )
      }

      func decode(
        tokenID: EdgeToolsToken.ID,
        position: Int
      ) async throws -> [Float] {
        try self.state.withLock { state in
          let inputIDs = try self.runtime.tensor(values: [Int64(tokenID)], shape: [1, 1])
          let position = try Int32(exactly: position).unwrapONNXInteger(
            name: TensorName.cachePosition
          )
          let cachePosition = try self.runtime.tensor(values: [position], shape: [1])
          let selfAttentionMask = try self.runtime.tensor(
            values: Self.selfAttentionMask(
              step: Int(position),
              maxLength: self.configuration.encoderMaxLength
            ),
            shape: [1, 1, 1, self.configuration.encoderMaxLength]
          )
          var outputs = try self.decoderSession.run(
            inputs: [
              TensorName.inputIDs: inputIDs,
              TensorName.cachePosition: cachePosition,
              TensorName.selfAttentionMask: selfAttentionMask,
              TensorName.crossAttentionMask: self.encoderOutputs.crossAttentionMask,
              TensorName.encoderProjectedK: self.encoderOutputs.encoderProjectedK,
              TensorName.encoderProjectedV: self.encoderOutputs.encoderProjectedV,
              TensorName.keyCache: state.cache.key,
              TensorName.valueCache: state.cache.value
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
          state.cache = Cache(key: updatedKey, value: updatedValue)
          return try logits.floatValues(count: self.configuration.vocabularySize)
        }
      }

      private static func selfAttentionMask(step: Int, maxLength: Int) -> [Float] {
        var values = [Float](repeating: -65500, count: maxLength)
        let allowedStart = max(0, maxLength - step - 1)
        for index in allowedStart..<maxLength {
          values[index] = 0
        }
        return values
      }
    }
  #endif

  #if canImport(COnnxRuntime)
    extension NeedleONNXEngine.ExecutionProvider {
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

    extension Optional where Wrapped: FixedWidthInteger {
      fileprivate func unwrapONNXInteger(name: String) throws -> Wrapped {
        guard let value = self else {
          throw EdgeToolsONNXRuntimeError(
            code: .integerConversionFailure,
            message: "Value for \(name) cannot be represented by the ONNX model's integer type."
          )
        }
        return value
      }
    }

    private enum TensorName {
      static let inputIDs = "input_ids"
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
#endif
