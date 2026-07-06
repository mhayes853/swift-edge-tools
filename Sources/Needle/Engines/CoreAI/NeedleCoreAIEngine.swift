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
        get async throws { try await self.task.cancellableValue }
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
        toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange =
          .unbounded(minimum: 0),
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

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderFunction: InferenceFunction
    private let decoderFunction: InferenceFunction
    private let tokenizer: any Tokenizer
    private let clock = ContinuousClock()
    private let decoderStates: Lock<DecoderStates?>

    public convenience init(
      modelDirectoryURL: URL,
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in },
      specializationOptions: SpecializationOptions = SpecializationOptions(
        preferredComputeUnitKind: .neuralEngine
      ),
      grammarEngine: (any Tokenizer) -> NeedleXGrammarEngine? = {
        NeedleXGrammarEngine(tokenizer: $0)
      }
    ) async throws {
      let tokenizer = try NeedleSPTokenizer(
        modelURL: modelDirectoryURL.appending(path: "tokenizer.model")
      )
      guard let grammarEngine = grammarEngine(tokenizer) else {
        throw NeedleCoreAIEngineError.failedToLoadGrammarEngine
      }

      var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
      editConfiguration(&configuration)

      async let encoderModel = AIModel(
        contentsOf: modelDirectoryURL.appending(path: "encoder.aimodel"),
        options: specializationOptions
      )
      async let decoderModel = AIModel(
        contentsOf: modelDirectoryURL.appending(path: "decoder.aimodel"),
        options: specializationOptions
      )
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
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
        State(grammarEngine: grammarEngine, matcherPool: NeedleGrammarMatcherPool())
      )
      self.configuration = configuration
      self.encoderFunction = try Self.loadFunction(named: FunctionName.main, from: encoderModel)
      let decoderFunction = try Self.loadFunction(named: FunctionName.main, from: decoderModel)
      self.decoderFunction = decoderFunction
      self.tokenizer = tokenizer
      self.decoderStates = Lock(
        try DecoderStates(descriptor: decoderFunction.descriptor, isPersistent: true)
      )
    }

    public func tokenize(prompt: NeedlePrompt) async throws -> [NeedleToken] {
      prompt.tokenized(using: self.tokenizer)
    }

    public func clearCaches() {
      self.state.withLock {
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
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
        let range = parameters.toolCallInvocationRange
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            tools: prompt.tools,
            range: range,
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
      let (encoderOutputs, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        configuration: configuration,
        processor: &processor
      )

      var decodeStates = try self.beginDecodeWithStates()
      defer { self.endDecodeWithStates(decodeStates) }
      var nextDecoderTokenId = configuration.decoderStartTokenId
      var detokenizer = StreamingDetokenizer(tokenizer: self.tokenizer)
      var generatedTokens = [NeedleToken]()
      var confidence = NeedleConfidenceState()
      var durationToFirstToken: Duration?

      while !matcher.isTerminated
        && !isStopped.load(ordering: .relaxed)
        && detokenizer.tokenIds.count < (parameters.maxTokens ?? .max)
        && generatedTokens.last?.id != self.tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        let inputIds = NDArray(tokenIds: [nextDecoderTokenId])
        let decoderLogits = try await self.decode(
          inputIds: inputIds,
          encoderOutputs: encoderOutputs,
          states: &decodeStates
        )
        var logits = try self.stepLogits(from: decoderLogits)
        let processedLogits = processor?.process(logits: &logits) ?? logits
        var maskedLogits = processedLogits
        applyBitmaskCoreAI(logits: &maskedLogits, mask: matcher.bitmask())
        try confidence.add(logits: maskedLogits)

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
      var metadata = NeedleMetadata()
      metadata.generationConfidence = confidence.mean
      metadata.perTokenConfidences = confidence.perTokenConfidences
      return NeedleEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: NeedleDecodeMetrics(
          tokens: generatedTokens.count,
          duration: generateStart.duration(to: self.clock.now) - finalDurationToFirstToken,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        metadata: metadata
      )
    }

    private func prefill(
      prompt: NeedlePrompt,
      configuration: NeedleModelConfiguration,
      processor: inout (any LogitsProcessor)?
    ) async throws -> (EncoderOutputs, NeedlePrefillMetrics) {
      let promptTokens = self.tokenizer.encode(text: prompt.formatted())
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw NeedleCoreAIEngineError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: configuration.encoderMaxLength
        )
      }

      let promptArray = NDArray(tokenIds: promptTokens)
      let prefillStart = self.clock.now
      processor?.prompt(promptArray)
      let encoderOutputs = try await self.encode(promptTokens: promptTokens)
      let metrics = NeedlePrefillMetrics(
        tokens: promptTokens.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      return (encoderOutputs, metrics)
    }

    private func encode(promptTokens: [NeedleToken.ID]) async throws -> EncoderOutputs {
      let inputIds = NDArray(tokenIds: promptTokens)
      var outputs = try await self.encoderFunction.run(inputs: [TensorName.inputIds: inputIds])
      guard
        let crossAttentionMask = outputs.remove(TensorName.crossAttentionMask)?.ndArray,
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
        let scalars = (0..<vocabularySize).map { Float(bfloat16Bits: view[offset + $0]) }
        return NDArray(scalars: scalars, shape: [1, scalars.count])
      default:
        throw NeedleCoreAIEngineError.unsupportedLogitsScalarType(logits.scalarType)
      }
    }

    private static func decodeConfiguration(
      from directory: URL
    ) throws -> NeedleModelConfiguration {
      guard let config = try NeedleModelConfiguration.decode(in: directory) else {
        throw NeedleCoreAIEngineError.failedToLoadConfiguration
      }
      return config
    }

    private static func loadFunction(
      named name: String,
      from model: AIModel
    ) throws -> InferenceFunction {
      guard let function = try model.loadFunction(named: name) else {
        throw NeedleCoreAIEngineError.failedToLoadFunction(name: name)
      }
      return function
    }
  }

  // MARK: - Decoder State

  @available(anyAppleOS 27.0, *)
  extension NeedleCoreAIEngine {
    private struct DecoderStates {
      var keyCache: NDArray
      var valueCache: NDArray
      var cacheOffset: NDArray
      let isPersistent: Bool

      init(descriptor: InferenceFunctionDescriptor, isPersistent: Bool) throws {
        guard
          let keyCacheDescriptor = descriptor.stateDescriptor(of: TensorName.keyCache),
          let valueCacheDescriptor = descriptor.stateDescriptor(of: TensorName.valueCache),
          let cacheOffsetDescriptor = descriptor.stateDescriptor(of: TensorName.cacheOffset)
        else {
          throw NeedleCoreAIEngineError.missingModelStateDescriptors
        }
        self.keyCache = try NDArray(descriptor: keyCacheDescriptor)
        self.valueCache = try NDArray(descriptor: valueCacheDescriptor)
        self.cacheOffset = try NDArray(descriptor: cacheOffsetDescriptor)
        self.isPersistent = isPersistent
      }

      mutating func reset() {
        self.keyCache.zero()
        self.valueCache.zero()
        self.cacheOffset.zero()
      }
    }

    private func beginDecodeWithStates() throws -> sending DecoderStates {
      try self.decoderStates.withLock { states in
        if var persistentStates = states {
          persistentStates.reset()
          states = nil
          return persistentStates
        } else {
          return try DecoderStates(
            descriptor: self.decoderFunction.descriptor,
            isPersistent: false
          )
        }
      }
    }

    private func endDecodeWithStates(_ states: sending DecoderStates) {
      guard states.isPersistent else { return }
      self.decoderStates.withLock { $0 = states }
    }

    private func decode(
      inputIds: NDArray,
      encoderOutputs: EncoderOutputs,
      states: inout DecoderStates
    ) async throws -> NDArray {
      var stateViews = InferenceFunction.MutableViews()
      stateViews.insert(&states.keyCache, for: TensorName.keyCache)
      stateViews.insert(&states.valueCache, for: TensorName.valueCache)
      stateViews.insert(&states.cacheOffset, for: TensorName.cacheOffset)

      var outputs = try await self.decoderFunction.run(
        inputs: [
          TensorName.inputIds: inputIds,
          TensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
          TensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
          TensorName.encoderProjectedV: encoderOutputs.encoderProjectedV
        ],
        states: stateViews
      )
      guard let logits = outputs.remove(TensorName.logits)?.ndArray else {
        throw NeedleCoreAIEngineError.missingModelOutputs
      }
      return logits
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
        let best = (0..<logits.shape[1])
          .max { lhs, rhs in
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

    public static let missingModelOutputs = Self(
      message: "CoreAI model did not return expected outputs."
    )

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

  // MARK: - Helpers

  @available(anyAppleOS 27.0, *)
  extension NDArray {
    fileprivate init(descriptor: InferenceValue.Descriptor) throws {
      guard case .ndArray(let descriptor) = descriptor else {
        throw NeedleCoreAIEngineError.missingModelStateDescriptors
      }
      self.init(descriptor: descriptor)
    }

    fileprivate init(tokenIds: some Sequence<NeedleToken.ID>) {
      let ids = tokenIds.map(Int32.init)
      self.init(scalars: ids, shape: [1, ids.count])
    }

    fileprivate mutating func zero() {
      var view = self.mutableRawView()
      let count = view.mutableBytes.byteCount
      var span = MutableSpan<UInt8>(mutableBytes: view.mutableBytes)
      for i in 0..<count {
        span[i] = 0
      }
    }
  }
#endif
