#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import Foundation
  import Tokenizers
  import Atomics

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIEngine: NeedleEngine, Sendable {
    public final class GenerationTask: NeedleEngineGenerationTask {
      private let task: Task<NeedleEngineGeneration, any Error>
      private let isStopped: ManagedAtomic<Bool>

      fileprivate init(
        task: sending Task<NeedleEngineGeneration, any Error>,
        isStopped: ManagedAtomic<Bool>
      ) {
        self.task = task
        self.isStopped = isStopped
      }

      public var value: NeedleEngineGeneration {
        get async throws {
          try await withTaskCancellationHandler {
            try await self.task.value
          } onCancel: {
            self.task.cancel()
          }
        }
      }

      public func stop() {
        self.isStopped.store(true, ordering: .relaxed)
      }
    }

    public struct GenerateParameters: NeedleEngineGenerateParameters {
      public static var `default`: Self { Self() }

      public var sampler: any Sampler
      public var processor: (any LogitsProcessor)?
      public var toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange
      public var maxTokens: Int?

      public init(
        sampler: any Sampler = ArgmaxSampler(),
        processor: (any LogitsProcessor)? = nil,
        toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange = .unbounded(
          minimum: 0
        ),
        maxTokens: Int? = 1024
      ) {
        self.sampler = sampler
        self.processor = processor
        self.toolCallInvocationRange = toolCallInvocationRange
        self.maxTokens = maxTokens
      }
    }

    private struct State {
      let grammarEngine: NeedleXGrammarEngine
      let matcherPool: NeedleGrammarMatcherPool
      var decoderState: DecoderState?
      var isDecoderStateInUse = false
    }

    private enum FunctionName {
      static let main = "main"
    }

    private enum TensorName {
      static let inputIds = "input_ids"
      static let crossAttentionMask = "cross_attention_mask"
      static let encoderProjectedK = "encoder_projected_k"
      static let encoderProjectedV = "encoder_projected_v"
      static let keyCache = "keyCache"
      static let valueCache = "valueCache"
      static let cacheOffset = "cacheOffset"
      static let logits = "logits"
    }

    private struct EncoderOutputs {
      let crossAttentionMask: NDArray
      let encoderProjectedK: NDArray
      let encoderProjectedV: NDArray
    }

    private actor DecoderState {
      private var keyCache: NDArray?
      private var valueCache: NDArray?
      private var cacheOffset: NDArray?

      init(descriptor: InferenceFunctionDescriptor) throws {
        guard let keyCacheDescriptor = descriptor.stateDescriptor(of: TensorName.keyCache),
          let valueCacheDescriptor = descriptor.stateDescriptor(of: TensorName.valueCache),
          let cacheOffsetDescriptor = descriptor.stateDescriptor(of: TensorName.cacheOffset)
        else {
          throw NeedleCoreAIEngineError.missingModelStateDescriptors
        }
        self.keyCache = try Self.array(from: keyCacheDescriptor)
        self.valueCache = try Self.array(from: valueCacheDescriptor)
        self.cacheOffset = try Self.array(from: cacheOffsetDescriptor)
      }

      func reset() throws {
        guard var keyCache = self.keyCache,
          var valueCache = self.valueCache,
          var cacheOffset = self.cacheOffset
        else {
          throw NeedleCoreAIEngineError.missingModelStateDescriptors
        }
        try Self.zero(&keyCache)
        try Self.zero(&valueCache)
        try Self.zeroCacheOffset(&cacheOffset)
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.cacheOffset = cacheOffset
      }

      func logits(
        function: InferenceFunction,
        inputIds: NDArray,
        encoderOutputs: EncoderOutputs
      ) async throws -> NDArray {
        guard var keyCache = self.keyCache,
          var valueCache = self.valueCache,
          var cacheOffset = self.cacheOffset
        else {
          throw NeedleCoreAIEngineError.missingModelStateDescriptors
        }

        var stateViews = InferenceFunction.MutableViews()
        stateViews.insert(&keyCache, for: TensorName.keyCache)
        stateViews.insert(&valueCache, for: TensorName.valueCache)
        stateViews.insert(&cacheOffset, for: TensorName.cacheOffset)

        var outputs = try await function.run(
          inputs: [
            TensorName.inputIds: inputIds,
            TensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
            TensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
            TensorName.encoderProjectedV: encoderOutputs.encoderProjectedV,
          ],
          states: stateViews
        )
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.cacheOffset = cacheOffset
        guard let logits = outputs.remove(TensorName.logits)?.ndArray else {
          throw NeedleCoreAIEngineError.missingModelOutputs
        }
        return logits
      }

      private static func array(from descriptor: InferenceValue.Descriptor) throws -> NDArray {
        guard case let .ndArray(descriptor) = descriptor else {
          throw NeedleCoreAIEngineError.missingModelStateDescriptors
        }
        return NDArray(descriptor: descriptor)
      }

      private static func zeroCacheOffset(_ array: inout NDArray) throws {
        guard array.scalarType == .int32 else {
          throw NeedleCoreAIEngineError.unsupportedStateScalarType(array.scalarType)
        }
        var view = array.mutableView(as: Int32.self)
        view.copyElements(fromContentsOf: [0])
      }

      private static func zero(_ array: inout NDArray) throws {
        let scalarCount = array.shape.reduce(1, *)
        switch array.scalarType {
        case .bfloat16:
          var rawView = array.mutableRawView()
          var view = MutableSpan<UInt16>(mutableBytes: rawView.mutableBytes)
          for index in 0..<scalarCount {
            view[index] = 0
          }
        case .int32:
          var view = array.mutableView(as: Int32.self)
          view.copyElements(fromContentsOf: repeatElement(0, count: scalarCount))
        default:
          throw NeedleCoreAIEngineError.unsupportedStateScalarType(array.scalarType)
        }
      }
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderFunction: InferenceFunction
    private let decoderFunction: InferenceFunction
    private let _tokenizer: any Tokenizer
    private let clock = ContinuousClock()

    public var tokenizer: any Tokenizer {
      self._tokenizer
    }

    public convenience init(
      modelDirectoryURL: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      grammarEngine: (any Tokenizer) -> NeedleXGrammarEngine? = {
        NeedleXGrammarEngine(tokenizer: $0)
      }
    ) async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: modelDirectoryURL.appending(path: "tokenizer.model"))
      guard let grammarEngine = grammarEngine(tokenizer) else {
        throw NeedleCoreAIEngineError.failedToLoadGrammarEngine
      }

      var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
      editConfiguration(&configuration)

      async let encoderModel = AIModel(contentsOf: modelDirectoryURL.appending(path: "encoder.aimodel"))
      async let decoderModel = AIModel(contentsOf: modelDirectoryURL.appending(path: "decoder.aimodel"))
      try self.init(
        encoderModel: try await encoderModel,
        decoderModel: try await decoderModel,
        tokenizer: tokenizer,
        configuration: configuration,
        grammarEngine: grammarEngine
      )
    }

    public convenience init(
      encoderModel: AIModel,
      decoderModel: AIModel,
      tokenizer: any Tokenizer,
      grammarEngine: sending NeedleXGrammarEngine
    ) throws {
      try self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: tokenizer,
        configuration: NeedleModelConfiguration(),
        grammarEngine: grammarEngine
      )
    }

    public init(
      encoderModel: AIModel,
      decoderModel: AIModel,
      tokenizer: any Tokenizer,
      configuration: NeedleModelConfiguration,
      grammarEngine: sending NeedleXGrammarEngine
    ) throws {
      self.state = Lock(
        State(
          grammarEngine: grammarEngine,
          matcherPool: NeedleGrammarMatcherPool(),
          decoderState: nil,
          isDecoderStateInUse: false
        )
      )
      self.configuration = configuration
      self.encoderFunction = try Self.loadFunction(named: FunctionName.main, from: encoderModel)
      self.decoderFunction = try Self.loadFunction(named: FunctionName.main, from: decoderModel)
      self._tokenizer = tokenizer
    }

    public func tokenize(prompt: NeedlePrompt) async throws -> [NeedleToken] {
      prompt.tokenized(using: self._tokenizer)
    }

    public func clearCaches() {
      self.state.withLock { state in
        state.matcherPool.clear()
        state.grammarEngine.clearCache()
        state.decoderState = nil
        state.isDecoderStateInUse = false
      }
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void
    ) throws -> GenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        nonisolated(unsafe) let parameters = parameters
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            tools: prompt.tools,
            range: parameters.toolCallInvocationRange,
            compilingWith: state.grammarEngine
          )
          matcher.reset()
          return matcher
        }
        return try await self.generate(
          prompt: prompt,
          parameters: parameters,
          onToken: onToken,
          matcher: matcher,
          configuration: self.configuration,
          isStopped: isStopped
        )
      }
      return GenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void,
      matcher: NeedleXGrammarEngine.Matcher,
      configuration: NeedleModelConfiguration,
      isStopped: ManagedAtomic<Bool>
    ) async throws -> NeedleEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      var processor = parameters.processor
      let generateStart = self.clock.now
      let (encoderOutputs, promptArray, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        configuration: configuration,
        processor: &processor
      )
      let decoderState = try self.acquireDecoderState()
      defer { self.releaseDecoderState() }
      try await decoderState.reset()

      var nextDecoderTokenId = configuration.decoderStartTokenId
      var detokenizer = StreamingDetokenizer(tokenizer: self._tokenizer)
      var generatedTokens = [NeedleToken]()
      var durationToFirstToken: Duration?

      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self._tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        let decoderLogits = try await self.decode(
          tokenId: nextDecoderTokenId,
          encoderOutputs: encoderOutputs,
          decoderState: decoderState
        )
        var logits = try self.stepLogits(from: decoderLogits)
        let processedLogits = processor?.process(logits: &logits) ?? logits
        var maskedLogits = processedLogits
        _ = applyBitmaskCoreAI(logits: &maskedLogits, mask: matcher.bitmask())

        let tokenId = parameters.sampler.sample(logits: maskedLogits)
        durationToFirstToken = durationToFirstToken ?? generateStart.duration(to: self.clock.now)

        let tokenString = detokenizer.decode(tokenId: tokenId)
        let token = NeedleToken(id: tokenId, stringValue: tokenString)
        generatedTokens.append(token)
        nextDecoderTokenId = tokenId
        guard matcher.accept(tokenId: token.id) else {
          throw NeedleCoreAIEngineError.grammarRejectedToken(token: token)
        }
        onToken(token)
        processor?.didSample(token: token)
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      _ = promptArray
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: generatedTokens.count,
          duration: generateStart.duration(to: self.clock.now) - finalDurationToFirstToken,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: isStopped.load(ordering: .relaxed),
        tokens: generatedTokens
      )
    }

    private func prefill(
      prompt: NeedlePrompt,
      configuration: NeedleModelConfiguration,
      processor: inout (any LogitsProcessor)?
    ) async throws -> (EncoderOutputs, NDArray, NeedlePrefillMetrics) {
      let promptTokens = self._tokenizer.encode(text: prompt.formatted())
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw NeedleCoreAIEngineError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: configuration.encoderMaxLength
        )
      }

      let promptArray = Self.array(tokenIds: promptTokens, shape: [1, promptTokens.count])
      let prefillStart = self.clock.now
      processor?.prompt(promptArray)
      let encoderOutputs = try await self.encode(promptTokens: promptTokens)
      let metrics = NeedlePrefillMetrics(
        tokens: promptTokens.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      return (encoderOutputs, promptArray, metrics)
    }

    private func encode(promptTokens: [NeedleToken.ID]) async throws -> EncoderOutputs {
      let inputIds = Self.array(tokenIds: promptTokens, shape: [1, promptTokens.count])
      var outputs = try await self.encoderFunction.run(inputs: [TensorName.inputIds: inputIds])
      guard let crossAttentionMask = outputs.remove(TensorName.crossAttentionMask)?.ndArray,
        let encoderProjectedK = outputs.remove(TensorName.encoderProjectedK)?.ndArray,
        let encoderProjectedV = outputs.remove(TensorName.encoderProjectedV)?.ndArray
      else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMask,
        encoderProjectedK: encoderProjectedK,
        encoderProjectedV: encoderProjectedV
      )
    }

    private func decode(
      tokenId: NeedleToken.ID,
      encoderOutputs: EncoderOutputs,
      decoderState: DecoderState
    ) async throws -> NDArray {
      let inputIds = Self.array(tokenIds: [tokenId], shape: [1, 1])
      return try await decoderState.logits(
        function: self.decoderFunction,
        inputIds: inputIds,
        encoderOutputs: encoderOutputs
      )
    }

    private func acquireDecoderState() throws -> DecoderState {
      try self.state.withLock { state in
        guard !state.isDecoderStateInUse else {
          throw NeedleCoreAIEngineError.decoderStateAlreadyInUse
        }
        if state.decoderState == nil {
          state.decoderState = try DecoderState(descriptor: self.decoderFunction.descriptor)
        }
        state.isDecoderStateInUse = true
        return state.decoderState!
      }
    }

    private func releaseDecoderState() {
      self.state.withLock { state in
        state.isDecoderStateInUse = false
      }
    }

    private func stepLogits(from logits: NDArray) throws -> NDArray {
      let stepIndex = logits.shape[1] - 1
      switch logits.scalarType {
      case .float32:
        let view = logits.view(as: Float.self)
        let scalars = (0..<logits.shape[2]).map { view[scalarAt: [0, stepIndex, $0]] }
        return NDArray(scalars: scalars, shape: [1, scalars.count])
      case .bfloat16:
        let shape = logits.shape
        let vocabularySize = shape[2]
        let offset = stepIndex * vocabularySize
        let view = Span<UInt16>(viewing: logits.rawView().bytes)
        let scalars = (0..<vocabularySize).map { index in
          Self.float32(fromBFloat16Bits: view[offset + index])
        }
        return NDArray(scalars: scalars, shape: [1, scalars.count])
      default:
        throw NeedleCoreAIEngineError.unsupportedLogitsScalarType(logits.scalarType)
      }
    }

    private static func array(tokenIds: [NeedleToken.ID], shape: [Int]) -> NDArray {
      NDArray(scalars: tokenIds.map(Int32.init), shape: shape)
    }

    private static func decodeConfiguration(from directory: URL) throws -> NeedleModelConfiguration {
      let decoder = JSONDecoder()
      let configurationURLs = [
        directory.appending(path: "configuration.json"),
        directory.appending(path: "config.json")
      ]
      for url in configurationURLs where FileManager.default.fileExists(atPath: url.path()) {
        return try decoder.decode(NeedleModelConfiguration.self, from: Data(contentsOf: url))
      }
      throw NeedleCoreAIEngineError.failedToLoadConfiguration
    }

    private static func loadFunction(named name: String, from model: AIModel) throws
      -> InferenceFunction
    {
      guard let function = try model.loadFunction(named: name) else {
        throw NeedleCoreAIEngineError.failedToLoadFunction(name: name)
      }
      return function
    }

    private static func float32(fromBFloat16Bits bits: UInt16) -> Float {
      Float(bitPattern: UInt32(bits) << 16)
    }
  }

  // MARK: - Sampler

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
    public protocol Sampler {
      func sample(logits: NDArray) -> NeedleToken.ID
    }

    public struct ArgmaxSampler: Sampler {
      public init() {}

      public func sample(logits: NDArray) -> NeedleToken.ID {
        let view = logits.view(as: Float.self)
        let best = (0..<logits.shape[1]).max { lhs, rhs in
          view[scalarAt: [0, lhs]] < view[scalarAt: [0, rhs]]
        }
        return best ?? 0
      }
    }
  }

  // MARK: - LogitsProcessor

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
    public protocol LogitsProcessor {
      mutating func prompt(_ prompt: NDArray)

      func process(logits: inout NDArray) -> NDArray

      mutating func didSample(token: NeedleToken)
    }
  }

  // MARK: - NeedleCoreAIEngineError

  @available(anyAppleOS 27.0, *)
  public struct NeedleCoreAIEngineError: Hashable, Error {
    public let message: String

    public static let failedToLoadConfiguration = Self(
      message: "Could not load model configuration."
    )

    public static let failedToLoadGrammarEngine = Self(
      message: "Could not load grammar engine."
    )

    public static func failedToLoadFunction(name: String) -> Self {
      Self(message: "Could not load CoreAI function named \(name).")
    }

    public static func grammarRejectedToken(token: NeedleToken) -> Self {
      Self(
        message:
          "Token (ID=\(token.id), VALUE=\(token.stringValue)) was rejected by the grammar matcher."
      )
    }

    public static func contextLengthExceeded(tokens: Int, maximum: Int) -> Self {
      Self(message: "Prompt token count (\(tokens)) exceeds the model context length (\(maximum)).")
    }

    public static let missingModelOutputs = Self(message: "CoreAI model did not return expected outputs.")

    public static let missingModelStateDescriptors = Self(
      message: "CoreAI model did not return expected state descriptors."
    )

    public static let decoderStateAlreadyInUse = Self(
      message: "CoreAI decoder state is already in use by another generation."
    )

    public static func unsupportedLogitsScalarType(_ scalarType: NDArray.ScalarType) -> Self {
      Self(message: "Unsupported logits scalar type: \(scalarType).")
    }

    public static func unsupportedStateScalarType(_ scalarType: NDArray.ScalarType) -> Self {
      Self(message: "Unsupported state scalar type: \(scalarType).")
    }
  }
#endif
