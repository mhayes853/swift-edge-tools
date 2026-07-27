#if System
  import SystemPackage
#endif

#if CoreML && canImport(CoreML)
  import Atomics
  @preconcurrency import CoreML
  import Foundation

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public final class NeedleCoreMLEngine: EdgeToolsEngine {
    public typealias Prompt = NeedlePrompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self { Self() }

      public var sampler: any EdgeToolsSampler<MLTensor>
      public var processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?

      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?

      public init(
        sampler: any EdgeToolsSampler<MLTensor> = CoreMLArgmaxSampler(),
        processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)? = nil,
        constraint: XGRGenerationConstraint = .tools,
        maxTokens: Int? = 1024
      ) {
        self.sampler = sampler
        self.processor = processor
        self.constraint = constraint
        self.maxTokens = maxTokens
      }
    }

    private struct State: ~Copyable {
      let grammarEngine: XGRCompiler
      let matcherPool: XGRToolCallMatcherPool
    }

    private struct EncoderOutputs {
      let crossAttentionMask: MLTensor
      let encoderProjectedK: MLTensor
      let encoderProjectedV: MLTensor
    }

    private struct DecoderCache: Sendable {
      var key: MLTensor
      var value: MLTensor

      init(key: MLTensor, value: MLTensor) {
        self.key = key
        self.value = value
      }

      init(configuration: NeedleModelConfiguration) {
        let shape = [
          configuration.decoderLayers,
          configuration.encoderMaxLength,
          configuration.attentionHeads,
          configuration.attentionHeadDimensions
        ]
        self.key = MLTensor(repeating: Float16.zero, shape: shape, scalarType: Float16.self)
        self.value = MLTensor(repeating: Float16.zero, shape: shape, scalarType: Float16.self)
      }
    }

    private let state: Lock<State>
    private let configuration: NeedleModelConfiguration
    private let encoderModel: EncoderModelActor
    private let decoderModel: DecoderModelActor
    private let tokenizer: any EdgeToolsXGRTokenizer
    private let clock = ContinuousClock()

    public convenience init(
      modelDirectoryURL: URL,
      modelConfiguration: MLModelConfiguration,
      editModelConfiguration: (inout MLModelConfiguration) -> Void = { _ in },
      editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: modelDirectoryURL)
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }

      var configuration = try Self.decodeConfiguration(from: modelDirectoryURL)
      editConfiguration(&configuration)

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
      try await self.init(
        encoderModel: encoderModel,
        decoderModel: decoderModel,
        tokenizer: tokenizer,
        configuration: configuration
      )
    }

    #if System
      public convenience init(
        modelDirectoryPath: FilePath,
        modelConfiguration: MLModelConfiguration,
        editModelConfiguration: (inout MLModelConfiguration) -> Void = { _ in },
        editConfiguration: (inout NeedleModelConfiguration) -> Void = { _ in }
      ) async throws {
        try await self.init(
          modelDirectoryURL: URL(filePath: modelDirectoryPath.string, directoryHint: .isDirectory),
          modelConfiguration: modelConfiguration,
          editModelConfiguration: editModelConfiguration,
          editConfiguration: editConfiguration
        )
      }
    #endif

    public init(
      encoderModel: sending MLModel,
      decoderModel: sending MLModel,
      tokenizer: sending any EdgeToolsXGRTokenizer,
      configuration: NeedleModelConfiguration
    ) throws {
      let grammarEngine = try XGRCompiler(
        tokenizerInfo: try tokenizer.tokenizerInfo(
          modelVocabularySize: configuration.vocabularySize
        )
      )
      self.state = Lock(
        State(
          grammarEngine: consume grammarEngine,
          matcherPool: XGRToolCallMatcherPool()
        )
      )
      self.configuration = configuration
      self.encoderModel = EncoderModelActor(model: encoderModel)
      self.decoderModel = DecoderModelActor(model: decoderModel)
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
        let toolsGrammar = try parameters.constraint.toolCallRange.map {
          try XGRGrammar.needle(tools: tools, range: $0)
        } ?? .universal
        let tokenizerInfo = self.state.withLock { $0.grammarEngine.tokenizerInfo }
        let grammar = try parameters.constraint.grammar(
          using: toolsGrammar,
          tokenizerInfo: tokenizerInfo
        )
        let matcher = try self.state.withLock { state in
          let matcher = try state.matcherPool.matcher(
            grammar: grammar,
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
          configuration: self.configuration,
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
      configuration: NeedleModelConfiguration,
      isStopped: ManagedAtomic<Bool>
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = consume matcher
      let sampler = parameters.sampler
      var processor = parameters.processor
      let generateStart = self.clock.now
      let (encoderOutputs, prefillMetrics) = try await self.prefill(
        prompt: prompt,
        tools: tools,
        configuration: configuration,
        processor: &processor
      )

      var cache = DecoderCache(configuration: configuration)
      var nextDecoderTokenId = configuration.decoderStartTokenId
      var loop = EdgeToolsGenerationLoop<NeedleToolCallParser>(
        matcher: consume matcher,
        tokenizer: self.tokenizer,
        channel: channel,
        isStopped: isStopped,
        maximumTokenCount: parameters.maxTokens,
        generateStart: generateStart
      )
      while let bitmask = try loop.nextBitmask() {
        let decoderLogits = try await self.decode(
          inputIds: MLTensor(tokenIds: [nextDecoderTokenId]),
          cachePosition: loop.generatedTokenCount,
          encoderOutputs: encoderOutputs,
          cache: &cache
        )
        var stepLogits = decoderLogits.squeezingShape(at: 1)
        let processedLogits = try await processor?.process(logits: &stepLogits) ?? stepLogits
        let maskedLogits = applyBitmaskCoreML(logits: processedLogits, mask: bitmask)
        let confidence = await tokenConfidenceCoreML(logits: maskedLogits)
        let tokenID = try await sampler.sample(logits: maskedLogits)
        let token = try loop.accept(tokenID: tokenID, confidence: confidence)
        nextDecoderTokenId = tokenID
        processor?.didSample(token: token)
      }

      return loop.finish(prefillMetrics: prefillMetrics)
    }

    private func prefill(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      configuration: NeedleModelConfiguration,
      processor: inout (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?
    ) async throws -> (EncoderOutputs, EdgeToolsPrefillMetrics) {
      let promptTokens = try self.tokenizer.encode(text: prompt.formatted(tools: tools))
      guard promptTokens.count <= configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: promptTokens.count,
          maximum: configuration.encoderMaxLength
        )
      }

      let promptTensor = MLTensor(tokenIds: promptTokens)
      let paddedPromptTensor = MLTensor(
        tokenIds: promptTokens,
        paddingTo: configuration.encoderMaxLength,
        padTokenId: configuration.padTokenId
      )
      let prefillStart = self.clock.now
      processor?.prompt(promptTensor)
      let encoderOutputs = try await self.runEncoder(inputIds: paddedPromptTensor)
      let metrics = EdgeToolsPrefillMetrics(
        tokens: promptTokens.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      return (encoderOutputs, metrics)
    }

    private func runEncoder(inputIds: MLTensor) async throws -> EncoderOutputs {
      var outputs = try await self.encoderModel.prediction(from: [NeedleExportTensorName.inputIDs: inputIds])
      guard
        let crossAttentionMask = outputs.removeValue(forKey: NeedleExportTensorName.crossAttentionMask),
        let encoderProjectedK = outputs.removeValue(forKey: NeedleExportTensorName.encoderProjectedK),
        let encoderProjectedV = outputs.removeValue(forKey: NeedleExportTensorName.encoderProjectedV)
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      return EncoderOutputs(
        crossAttentionMask: crossAttentionMask,
        encoderProjectedK: encoderProjectedK,
        encoderProjectedV: encoderProjectedV
      )
    }

    private func decode(
      inputIds: MLTensor,
      cachePosition: Int,
      encoderOutputs: EncoderOutputs,
      cache: inout DecoderCache
    ) async throws -> MLTensor {
      let inputs = [
        NeedleExportTensorName.inputIDs: inputIds,
        NeedleExportTensorName.cachePosition: MLTensor(shape: [1], scalars: [Int32(cachePosition)]),
        NeedleExportTensorName.selfAttentionMask: Self.selfAttentionMask(
          step: cachePosition,
          maxLength: self.configuration.encoderMaxLength
        ),
        NeedleExportTensorName.crossAttentionMask: encoderOutputs.crossAttentionMask,
        NeedleExportTensorName.encoderProjectedK: encoderOutputs.encoderProjectedK,
        NeedleExportTensorName.encoderProjectedV: encoderOutputs.encoderProjectedV,
        NeedleExportTensorName.keyCache: cache.key,
        NeedleExportTensorName.valueCache: cache.value
      ]
      let outputs = try await self.decoderModel.prediction(from: inputs)
      guard
        let logits = outputs[NeedleExportTensorName.logits],
        let updatedKey = outputs[NeedleExportTensorName.updatedKeyCache],
        let updatedValue = outputs[NeedleExportTensorName.updatedValueCache]
      else {
        throw EdgeToolsError.missingModelOutputs
      }
      cache = DecoderCache(key: updatedKey, value: updatedValue)
      return logits
    }

  }

  // MARK: - Model Loading

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleCoreMLEngine {
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
  extension NeedleCoreMLEngine {
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

      func prediction(from inputs: [String: MLTensor]) async throws -> [String: MLTensor] {
        try await self.model.prediction(from: inputs)
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
  extension NeedleCoreMLEngine {
    private static func selfAttentionMask(step: Int, maxLength: Int) -> MLTensor {
      var scalars = Array(repeating: Float16(-65500), count: maxLength)
      let allowedStart = max(0, maxLength - step - 1)
      for index in allowedStart..<maxLength {
        scalars[index] = 0
      }
      return MLTensor(shape: [1, 1, 1, maxLength], scalars: scalars)
    }
  }

  private enum ModelName {
    static let encoder = "encoder"
    static let decoder = "decoder"
  }

#endif
