#if ONNXCore
  import Atomics
  import Foundation

  #if ONNX && canImport(COnnxRuntime)
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

  #if ONNX && canImport(COnnxRuntime)
    public final class NeedleONNXRuntimeBackend: NeedleONNXBackend {
      public let configuration: NeedleModelConfiguration

      private let runtime: ONNXRuntime
      private let encoderSession: ONNXRuntimeSession
      private let decoderSession: ONNXRuntimeSession

      public init(
        modelDirectoryURL: URL,
        runtimeConfiguration: NeedleONNXEngine.RuntimeConfiguration =
          NeedleONNXEngine.RuntimeConfiguration(),
        configuration: NeedleModelConfiguration
      ) throws {
        let runtime = try ONNXRuntime()
        self.configuration = configuration
        self.runtime = runtime
        self.encoderSession = try runtime.session(
          modelURL: modelDirectoryURL.appending(path: "encoder.onnx"),
          configuration: runtimeConfiguration,
          expectedInputNames: [TensorName.inputIDs],
          expectedOutputNames: [
            TensorName.crossAttentionMask,
            TensorName.encoderProjectedK,
            TensorName.encoderProjectedV
          ]
        )
        self.decoderSession = try runtime.session(
          modelURL: modelDirectoryURL.appending(path: "decoder.onnx"),
          configuration: runtimeConfiguration,
          expectedInputNames: [
            TensorName.inputIDs,
            TensorName.cachePosition,
            TensorName.selfAttentionMask,
            TensorName.crossAttentionMask,
            TensorName.encoderProjectedK,
            TensorName.encoderProjectedV,
            TensorName.keyCache,
            TensorName.valueCache
          ],
          expectedOutputNames: [
            TensorName.logits,
            TensorName.updatedKeyCache,
            TensorName.updatedValueCache
          ]
        )
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
        let inputIDs = try self.runtime.int64Tensor(
          paddedTokens,
          shape: [1, self.configuration.encoderMaxLength]
        )
        var outputs = try self.encoderSession.run(
          inputs: [(TensorName.inputIDs, inputIDs)],
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
        return try NeedleONNXRuntimeGeneration(
          runtime: self.runtime,
          decoderSession: self.decoderSession,
          configuration: self.configuration,
          encoderOutputs: NeedleONNXRuntimeGeneration.EncoderOutputs(
            crossAttentionMask: crossAttentionMask,
            encoderProjectedK: encoderProjectedK,
            encoderProjectedV: encoderProjectedV
          )
        )
      }
    }

    private final class NeedleONNXRuntimeGeneration: NeedleONNXGeneration {
      struct EncoderOutputs: Sendable {
        let crossAttentionMask: ONNXRuntimeTensor
        let encoderProjectedK: ONNXRuntimeTensor
        let encoderProjectedV: ONNXRuntimeTensor
      }

      private struct Cache: Sendable {
        var key: ONNXRuntimeTensor
        var value: ONNXRuntimeTensor
      }

      private struct State: ~Copyable {
        var cache: Cache
      }

      private let runtime: ONNXRuntime
      private let decoderSession: ONNXRuntimeSession
      private let configuration: NeedleModelConfiguration
      private let encoderOutputs: EncoderOutputs
      private let state: Lock<State>

      init(
        runtime: ONNXRuntime,
        decoderSession: ONNXRuntimeSession,
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
              key: try runtime.floatTensor(repeating: 0, shape: cacheShape),
              value: try runtime.floatTensor(repeating: 0, shape: cacheShape)
            )
          )
        )
      }

      func decode(
        tokenID: EdgeToolsToken.ID,
        position: Int
      ) async throws -> [Float] {
        try self.state.withLock { state in
          let inputIDs = try self.runtime.int64Tensor([Int64(tokenID)], shape: [1, 1])
          let position = try Int32(exactly: position)
            .unwrapONNXInteger(
              name: TensorName.cachePosition
            )
          let cachePosition = try self.runtime.int32Tensor([position], shape: [1])
          let selfAttentionMask = try self.runtime.floatTensor(
            Self.selfAttentionMask(
              step: Int(position),
              maxLength: self.configuration.encoderMaxLength
            ),
            shape: [1, 1, 1, self.configuration.encoderMaxLength]
          )
          var outputs = try self.decoderSession.run(
            inputs: [
              (TensorName.inputIDs, inputIDs),
              (TensorName.cachePosition, cachePosition),
              (TensorName.selfAttentionMask, selfAttentionMask),
              (TensorName.crossAttentionMask, self.encoderOutputs.crossAttentionMask),
              (TensorName.encoderProjectedK, self.encoderOutputs.encoderProjectedK),
              (TensorName.encoderProjectedV, self.encoderOutputs.encoderProjectedV),
              (TensorName.keyCache, state.cache.key),
              (TensorName.valueCache, state.cache.value)
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

    public struct EdgeToolsONNXRuntimeError: Hashable, Sendable, Error {
      public struct Code: RawRepresentable, Hashable, Sendable {
        public let rawValue: Int

        public init(rawValue: Int) {
          self.rawValue = rawValue
        }
      }

      public let code: Code
      public let message: String

      public init(code: Code, message: String) {
        self.code = code
        self.message = message
      }
    }

    private final class ONNXRuntime: @unchecked Sendable {
      let api: UnsafePointer<OrtApi>
      let environment: OpaquePointer
      let allocator: UnsafeMutablePointer<OrtAllocator>

      init() throws {
        guard
          let apiBase = OrtGetApiBase(),
          let api = apiBase.pointee.GetApi(UInt32(ORT_API_VERSION))
        else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime does not support API version \(ORT_API_VERSION)."
          )
        }
        self.api = api

        var environment: OpaquePointer?
        try Self.check(
          api: api,
          status: api.pointee.CreateEnv(
            ORT_LOGGING_LEVEL_WARNING,
            "swift-edge-tools",
            &environment
          )
        )
        guard let environment else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime did not create an environment."
          )
        }
        self.environment = environment

        var allocator: UnsafeMutablePointer<OrtAllocator>?
        try Self.check(
          api: api,
          status: api.pointee.GetAllocatorWithDefaultOptions(&allocator)
        )
        guard let allocator else {
          api.pointee.ReleaseEnv(environment)
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime did not provide its default allocator."
          )
        }
        self.allocator = allocator
      }

      deinit {
        self.api.pointee.ReleaseEnv(self.environment)
      }

      func session(
        modelURL: URL,
        configuration: NeedleONNXEngine.RuntimeConfiguration,
        expectedInputNames: [String],
        expectedOutputNames: [String]
      ) throws -> ONNXRuntimeSession {
        var options: OpaquePointer?
        try Self.check(api: self.api, status: self.api.pointee.CreateSessionOptions(&options))
        guard let options else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime did not create session options."
          )
        }
        defer { self.api.pointee.ReleaseSessionOptions(options) }

        try Self.check(
          api: self.api,
          status: self.api.pointee.SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL)
        )
        if case .coreML(let computeUnits) = configuration.executionProvider {
          try self.appendCoreML(to: options, computeUnits: computeUnits)
        }

        var session: OpaquePointer?
        try modelURL.path()
          .withCString { path in
            try Self.check(
              api: self.api,
              status: self.api.pointee.CreateSession(
                self.environment,
                path,
                options,
                &session
              )
            )
          }
        guard let session else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime did not create a session for \(modelURL.path())."
          )
        }
        let runtimeSession = ONNXRuntimeSession(
          api: self.api,
          allocator: self.allocator,
          session: session
        )
        try runtimeSession.validateSignature(
          inputNames: expectedInputNames,
          outputNames: expectedOutputNames
        )
        return runtimeSession
      }

      func int64Tensor(_ values: [Int64], shape: [Int]) throws -> ONNXRuntimeTensor {
        try self.tensor(
          values: values,
          shape: shape,
          elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        )
      }

      func int32Tensor(_ values: [Int32], shape: [Int]) throws -> ONNXRuntimeTensor {
        try self.tensor(
          values: values,
          shape: shape,
          elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        )
      }

      func floatTensor(_ values: [Float], shape: [Int]) throws -> ONNXRuntimeTensor {
        try self.tensor(
          values: values,
          shape: shape,
          elementType: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        )
      }

      func floatTensor(repeating value: Float, shape: [Int]) throws -> ONNXRuntimeTensor {
        let count = shape.reduce(1, *)
        return try self.floatTensor([Float](repeating: value, count: count), shape: shape)
      }

      private func tensor<Element>(
        values: [Element],
        shape: [Int],
        elementType: ONNXTensorElementDataType
      ) throws -> ONNXRuntimeTensor {
        let dimensions = shape.map(Int64.init)
        let expectedCount = shape.reduce(1, *)
        guard values.count == expectedCount else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "Tensor shape \(shape) requires \(expectedCount) values, got \(values.count)."
          )
        }

        var tensor: OpaquePointer?
        try dimensions.withUnsafeBufferPointer { dimensions in
          try Self.check(
            api: self.api,
            status: self.api.pointee.CreateTensorAsOrtValue(
              self.allocator,
              dimensions.baseAddress,
              dimensions.count,
              elementType,
              &tensor
            )
          )
        }
        guard let tensor else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime did not create a tensor."
          )
        }

        do {
          var bytes: UnsafeMutableRawPointer?
          try Self.check(
            api: self.api,
            status: self.api.pointee.GetTensorMutableData(tensor, &bytes)
          )
          guard let bytes else {
            throw EdgeToolsONNXRuntimeError(
              code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
              message: "ONNX Runtime tensor did not provide mutable storage."
            )
          }
          values.withUnsafeBufferPointer { source in
            guard let sourceAddress = source.baseAddress else { return }
            bytes.copyMemory(
              from: sourceAddress,
              byteCount: values.count * MemoryLayout<Element>.stride
            )
          }
          return ONNXRuntimeTensor(api: self.api, tensor: tensor)
        } catch {
          self.api.pointee.ReleaseValue(tensor)
          throw error
        }
      }

      private func appendCoreML(
        to options: OpaquePointer,
        computeUnits: NeedleONNXEngine.CoreMLComputeUnits
      ) throws {
        let keys = [
          "MLComputeUnits",
          "ModelFormat",
          "RequireStaticInputShapes",
          "EnableOnSubgraphs"
        ]
        let values = [computeUnits.rawValue, "MLProgram", "1", "1"]
        try withCopiedCStringPointerBuffer(keys) { keyPointers in
          try withCopiedCStringPointerBuffer(values) { valuePointers in
            try "CoreML"
              .withCString { providerName in
                try Self.check(
                  api: self.api,
                  status: self.api.pointee.SessionOptionsAppendExecutionProvider(
                    options,
                    providerName,
                    keyPointers.baseAddress,
                    valuePointers.baseAddress,
                    keyPointers.count
                  )
                )
              }
          }
        }
      }

      static func check(api: UnsafePointer<OrtApi>, status: OpaquePointer?) throws {
        guard let status else { return }
        defer { api.pointee.ReleaseStatus(status) }
        let code = api.pointee.GetErrorCode(status)
        let message =
          api.pointee.GetErrorMessage(status).map(String.init(cString:))
          ?? "Unknown ONNX Runtime error."
        throw EdgeToolsONNXRuntimeError(
          code: EdgeToolsONNXRuntimeError.Code(rawValue: Int(code.rawValue)),
          message: message
        )
      }
    }

    // Safe because ONNX Runtime environments and sessions support concurrent inference, and this
    // wrapper never mutates or replaces the session pointer after initialization.
    private final class ONNXRuntimeSession: @unchecked Sendable {
      private let api: UnsafePointer<OrtApi>
      private let allocator: UnsafeMutablePointer<OrtAllocator>
      private let session: OpaquePointer

      init(
        api: UnsafePointer<OrtApi>,
        allocator: UnsafeMutablePointer<OrtAllocator>,
        session: OpaquePointer
      ) {
        self.api = api
        self.allocator = allocator
        self.session = session
      }

      deinit {
        self.api.pointee.ReleaseSession(self.session)
      }

      func validateSignature(
        inputNames expectedInputNames: [String],
        outputNames expectedOutputNames: [String]
      ) throws {
        let inputNames = try self.names(
          count: self.api.pointee.SessionGetInputCount,
          name: self.api.pointee.SessionGetInputName
        )
        let outputNames = try self.names(
          count: self.api.pointee.SessionGetOutputCount,
          name: self.api.pointee.SessionGetOutputName
        )
        guard inputNames == expectedInputNames, outputNames == expectedOutputNames else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message:
              "Invalid ONNX model signature. Expected inputs \(expectedInputNames) and outputs \(expectedOutputNames), got inputs \(inputNames) and outputs \(outputNames)."
          )
        }
      }

      func run(
        inputs: [(String, ONNXRuntimeTensor)],
        outputNames: [String]
      ) throws -> [String: ONNXRuntimeTensor] {
        let inputValues = inputs.map { Optional($0.1.tensor) }
        var outputValues = [OpaquePointer?](repeating: nil, count: outputNames.count)
        try withCopiedCStringPointerBuffer(inputs.map(\.0)) { inputNamePointers in
          try withCopiedCStringPointerBuffer(outputNames) { outputNamePointers in
            try inputValues.withUnsafeBufferPointer { inputValues in
              try outputValues.withUnsafeMutableBufferPointer { outputValues in
                try ONNXRuntime.check(
                  api: self.api,
                  status: self.api.pointee.Run(
                    self.session,
                    nil,
                    inputNamePointers.baseAddress,
                    inputValues.baseAddress,
                    inputValues.count,
                    outputNamePointers.baseAddress,
                    outputNamePointers.count,
                    outputValues.baseAddress
                  )
                )
              }
            }
          }
        }

        return try Dictionary(
          uniqueKeysWithValues: zip(outputNames, outputValues)
            .map { name, pointer in
              guard let pointer else {
                throw EdgeToolsONNXRuntimeError(
                  code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
                  message: "ONNX Runtime did not return output \(name)."
                )
              }
              return (name, ONNXRuntimeTensor(api: self.api, tensor: pointer))
            }
        )
      }

      private func names(
        count getCount: (OpaquePointer?, UnsafeMutablePointer<Int>?) -> OpaquePointer?,
        name getName: (
          OpaquePointer?,
          Int,
          UnsafeMutablePointer<OrtAllocator>?,
          UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
        ) -> OpaquePointer?
      ) throws -> [String] {
        var count = 0
        try ONNXRuntime.check(api: self.api, status: getCount(self.session, &count))
        return try (0..<count)
          .map { index in
            var name: UnsafeMutablePointer<CChar>?
            try ONNXRuntime.check(
              api: self.api,
              status: getName(self.session, index, self.allocator, &name)
            )
            guard let name else {
              throw EdgeToolsONNXRuntimeError(
                code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
                message: "ONNX Runtime did not return a model input or output name."
              )
            }
            defer { self.allocator.pointee.Free(self.allocator, name) }
            return String(cString: name)
          }
      }
    }

    // Safe because tensor storage is fully initialized before publication and this wrapper only
    // provides immutable reads. Every generated tensor has independent ONNX Runtime-owned storage.
    private final class ONNXRuntimeTensor: @unchecked Sendable {
      private let api: UnsafePointer<OrtApi>
      fileprivate let tensor: OpaquePointer

      init(api: UnsafePointer<OrtApi>, tensor: OpaquePointer) {
        self.api = api
        self.tensor = tensor
      }

      deinit {
        self.api.pointee.ReleaseValue(self.tensor)
      }

      func floatValues(count: Int) throws -> [Float] {
        var shapeInfo: OpaquePointer?
        try ONNXRuntime.check(
          api: self.api,
          status: self.api.pointee.GetTensorTypeAndShape(self.tensor, &shapeInfo)
        )
        guard let shapeInfo else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime did not return tensor type and shape information."
          )
        }
        defer { self.api.pointee.ReleaseTensorTypeAndShapeInfo(shapeInfo) }

        var elementType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
        try ONNXRuntime.check(
          api: self.api,
          status: self.api.pointee.GetTensorElementType(shapeInfo, &elementType)
        )
        var elementCount = 0
        try ONNXRuntime.check(
          api: self.api,
          status: self.api.pointee.GetTensorShapeElementCount(shapeInfo, &elementCount)
        )
        guard elementType == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, elementCount == count else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message:
              "Expected a Float32 tensor with \(count) values, got element type \(elementType.rawValue) with \(elementCount) values."
          )
        }

        var bytes: UnsafeMutableRawPointer?
        try ONNXRuntime.check(
          api: self.api,
          status: self.api.pointee.GetTensorMutableData(self.tensor, &bytes)
        )
        guard let bytes else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
            message: "ONNX Runtime tensor did not provide Float32 storage."
          )
        }
        return Array(
          UnsafeBufferPointer(start: bytes.assumingMemoryBound(to: Float.self), count: count)
        )
      }
    }
  #endif

  #if ONNX && canImport(COnnxRuntime)
    extension Optional where Wrapped: FixedWidthInteger {
      fileprivate func unwrapONNXInteger(name: String) throws -> Wrapped {
        guard let value = self else {
          throw EdgeToolsONNXRuntimeError(
            code: EdgeToolsONNXRuntimeError.Code(rawValue: -1),
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
