#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN
  import Observation

  #if canImport(Tokenizers)
    import Tokenizers
  #endif

  #if canImport(CoreImage) && canImport(MLXVLM)
    import CoreImage
    import Foundation
    import MLXVLM
  #endif

  #if canImport(MLXHuggingFace) && canImport(Tokenizers)
    import MLXHuggingFace
  #endif

  // MARK: - MLXModelProfile

  public protocol MLXModelProfile: SendableMetatype {
    associatedtype Prompt: Sendable
    associatedtype GenerateParameters: MLXGenerateParameters
    associatedtype GenerationParser: EdgeToolsGenerationParser
    associatedtype GrammarEngine: EdgeToolsGrammarEngine

    static var extraStopTokens: Set<String> { get }

    static func grammarEngine(
      tokenizer: any EdgeToolsTokenizer,
      vocabularySize: Int,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> GrammarEngine

    static func grammar(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      grammarEngine: borrowing GrammarEngine
    ) throws -> GrammarEngine.Grammar

    static func prepare(
      prompt: inout Prompt,
      tools: [EdgeToolDefinition],
      parser: inout GenerationParser
    )

    static func templateContext(prompt: Prompt) -> [String: any Sendable]?

    static nonisolated(nonsending) func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput

    static func defaultSampling(
      prompt: Prompt,
      parameters: GenerateParameters
    ) -> EdgeToolsFusedSamplingParameters?
  }

  public protocol MLXLLMModelProfile: MLXModelProfile {}

  #if canImport(CoreImage) && canImport(MLXVLM)
    public protocol MLXVLMModelProfile: MLXModelProfile {}
  #endif

  extension MLXModelProfile
  where
    GenerateParameters: EdgeToolsConstrainedGenerateParameters,
    GenerateParameters.Constraint.Grammar == GrammarEngine.Grammar,
    GenerateParameters.Constraint.Context == GrammarEngine
  {
    public static func constrainedGrammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      grammarEngine: borrowing GrammarEngine,
      toolCallGrammar: (GrammarToolCallRange) throws -> GrammarEngine.Grammar
    ) throws -> GrammarEngine.Grammar {
      let constraint = parameters.constraint
      let grammar = try constraint.toolCallRange.map(toolCallGrammar)
      return try constraint.grammar(toolCallGrammar: grammar, context: grammarEngine)
    }
  }

  extension MLXModelProfile {
    public static var extraStopTokens: Set<String> { [] }

    public static func defaultSampling(
      prompt: Prompt,
      parameters: GenerateParameters
    ) -> EdgeToolsFusedSamplingParameters? {
      nil
    }

    public static func prepare(
      prompt: inout Prompt,
      tools: [EdgeToolDefinition],
      parser: inout GenerationParser
    ) {}

    public static func templateContext(prompt: Prompt) -> [String: any Sendable]? {
      nil
    }
  }

  public struct MLXEngineError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let emptyInput = Self(rawValue: "empty-input")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - MLXGenerateParameters

  public protocol MLXGenerateParameters: EdgeToolsEngineGenerateParameters {
    var sampler: (any LogitSampler)? { get }
    var samplingOverrides: EdgeToolsFusedSamplingOverrides { get }
    var processor: (any LogitProcessor)? { get }
    var kvCacheQuantizationBits: Int? { get }
    var kvCacheQuantizationGroupSize: Int { get }
    var quantizedKVStart: Int { get }
    var synchronizeStreamForMemorySnapshots: Bool { get }
  }

  extension MLXGenerateParameters {
    public var samplingOverrides: EdgeToolsFusedSamplingOverrides {
      EdgeToolsFusedSamplingOverrides()
    }
  }

  // MARK: - DefaultMLXGenerateParameters

  #if XGrammar
    public struct DefaultMLXGenerateParameters:
      MLXGenerateParameters,
      EdgeToolsConstrainedGenerateParameters
    {
      public static var `default`: Self { Self() }

      public var sampler: (any LogitSampler)?
      public var samplingOverrides: EdgeToolsFusedSamplingOverrides
      public var processor: (any LogitProcessor)?
      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int
      public var quantizedKVStart: Int
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: (any LogitSampler)? = nil,
        samplingOverrides: EdgeToolsFusedSamplingOverrides = EdgeToolsFusedSamplingOverrides(),
        processor: (any LogitProcessor)? = nil,
        constraint: XGRGenerationConstraint = .unconstrained,
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self.sampler = sampler
        self.samplingOverrides = samplingOverrides
        self.processor = processor
        self.constraint = constraint
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
      }
    }

    // MARK: - MLXModelProfile + XGrammar

    extension MLXModelProfile where GrammarEngine == XGrammarEngine {
      public static func grammarEngine(
        tokenizer: any EdgeToolsTokenizer,
        vocabularySize: Int,
        stopTokenIds: Set<EdgeToolsToken.ID>
      ) throws -> XGrammarEngine {
        guard let tokenizer = tokenizer as? any XGRTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        return try XGrammarEngine(
          tokenizerInfo: tokenizer.tokenizerInfo(
            modelVocabularySize: vocabularySize,
            extraStopTokenIds: stopTokenIds
          )
        )
      }
    }
  #endif

  // MARK: - MLXContextParameters

  public struct MLXContextParameters: Hashable, Sendable {
    public var transcript: EdgeToolsTranscript
    public var reasoningEffort: EdgeToolsReasoningEffort

    public init(
      transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) {
      self.transcript = transcript
      self.reasoningEffort = reasoningEffort
    }
  }

  // MARK: - MLXContext

  public final class MLXContext<Profile: MLXModelProfile>: Identifiable, Sendable
  where Profile.Prompt == EdgeToolsTranscript {
    private struct State {
      var transcript: EdgeToolsTranscript
      var reasoningEffort: EdgeToolsReasoningEffort
      var isResponding = false
      var revision = 0
      var model: MLXModelState<Profile>?

      mutating func finish(
        generation: EdgeToolsEngineGeneration?,
        revision: Int,
        model: sending MLXModelState<Profile>,
        makeModelState: @Sendable () -> sending MLXModelState<Profile>
      ) {
        self.model = self.revision == revision ? model : makeModelState()
        if let generation {
          self.transcript.messages.append(.init(generation: generation))
          self.revision += 1
        }
        self.isResponding = false
      }
    }

    struct GenerationSnapshot {
      let transcript: EdgeToolsTranscript
      let revision: Int
      let model: MLXModelState<Profile>
    }

    private let state: Lock<State>
    private let makeModelState: @Sendable () -> sending MLXModelState<Profile>
    private let observationRegistrar = _ObservationRegistrar()

    public var transcript: EdgeToolsTranscript {
      get {
        self.observationRegistrar.access(self, keyPath: \.transcript)
        return self.state.withBorrowedLock { $0.transcript }
      }
      set {
        self.state.withLock { state in
          self.observationRegistrar.withMutation(of: self, keyPath: \.transcript) {
            state.transcript = newValue
            state.revision += 1
          }
        }
      }
    }

    public var reasoningEffort: EdgeToolsReasoningEffort {
      get {
        self.observationRegistrar.access(self, keyPath: \.reasoningEffort)
        return self.state.withBorrowedLock { $0.reasoningEffort }
      }
      set {
        self.state.withLock { state in
          self.observationRegistrar.withMutation(of: self, keyPath: \.reasoningEffort) {
            state.reasoningEffort = newValue
            state.revision += 1
          }
        }
      }
    }

    public var isResponding: Bool {
      self.observationRegistrar.access(self, keyPath: \.isResponding)
      return self.state.withBorrowedLock { $0.isResponding }
    }

    init(
      parameters: MLXContextParameters,
      model: sending MLXModelState<Profile>,
      makeModelState: @escaping @Sendable () -> sending MLXModelState<Profile>
    ) {
      self.state = Lock(
        State(
          transcript: parameters.transcript,
          reasoningEffort: parameters.reasoningEffort,
          model: model
        )
      )
      self.makeModelState = makeModelState
    }

    func transcript(
      appending message: EdgeToolsTranscript.UserMessage
    ) -> EdgeToolsTranscript {
      self.state.withBorrowedLock { state in
        var transcript = state.transcript
        transcript.messages.append(.user(message))
        transcript.reasoningEffort = state.reasoningEffort
        return transcript
      }
    }

    func begin(appending message: EdgeToolsTranscript.UserMessage) throws -> GenerationSnapshot {
      try self.state.withLock { state in
        guard !state.isResponding, let model = state.model.take() else {
          throw EdgeToolsError.contextInUse
        }
        var transcript = state.transcript
        transcript.messages.append(.user(message))
        var snapshot: GenerationSnapshot!
        self.observationRegistrar.withMutation(of: self, keyPath: \.transcript) {
          self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
            state.transcript = transcript
            state.revision += 1
            state.isResponding = true
            transcript.reasoningEffort = state.reasoningEffort
            snapshot = GenerationSnapshot(
              transcript: transcript,
              revision: state.revision,
              model: model
            )
          }
        }
        return snapshot
      }
    }

    func begin() throws -> GenerationSnapshot {
      try self.state.withLock { state in
        guard !state.isResponding, let model = state.model.take() else {
          throw EdgeToolsError.contextInUse
        }
        var snapshot: GenerationSnapshot!
        self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
          state.isResponding = true
          var transcript = state.transcript
          transcript.reasoningEffort = state.reasoningEffort
          snapshot = GenerationSnapshot(
            transcript: transcript,
            revision: state.revision,
            model: model
          )
        }
        return snapshot
      }
    }

    func finish(
      generation: EdgeToolsEngineGeneration?,
      revision: Int,
      model: sending MLXModelState<Profile>
    ) {
      // The compiler cannot track a `sending` parameter through the lock closure. The context
      // removed its only reference at `begin`, and this method stores the returned value once.
      nonisolated(unsafe) let model = model
      self.state.withLock { state in
        state.finish(
          generation: generation,
          revision: revision,
          model: model,
          makeModelState: self.makeModelState
        )
      }
      self.observationRegistrar.withMutation(of: self, keyPath: \.transcript) {
      }
      self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
      }
    }
  }

  extension MLXContext: Observable {}

  // MARK: - Prompt Conversion

  #if canImport(Tokenizers)
    extension MLXLLMModelProfile where Prompt == EdgeToolsTranscript {
      public static nonisolated(nonsending) func input(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        processor: (any UserInputProcessor)?
      ) async throws -> LMInput {
        guard let tokenizer = tokenizer as? TransformersTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let tokenIds = try tokenizer.base.applyChatTemplate(
          messages: try prompt.mlxMessages(),
          tools: tools.mlxToolSpecs,
          additionalContext: Self.templateContext(prompt: prompt)
        )
        return LMInput(tokens: MLXArray(tokenIds))
      }
    }

    extension EdgeToolsTranscript {
      fileprivate func mlxMessages() throws -> [MLXLMCommon.Message] {
        try self.messages.map { try $0.mlxMessage() }
      }
    }

    extension EdgeToolsTranscript.Message {
      public func mlxMessage() throws -> MLXLMCommon.Message {
        switch self {
        case .system(let message):
          ["role": "system", "content": message.content]
        case .user(let message):
          ["role": "user", "content": message.content]
        case .assistant(let message):
          self.mlxAssistantMessage(parts: message.parts)
        case .tool(let message):
          [
            "role": "tool",
            "content": String(decoding: try Self.encode(message.response), as: UTF8.self),
            "name": message.name
          ]
        }
      }

      private func mlxAssistantMessage(parts: [EdgeToolsGenerationPart]) -> MLXLMCommon.Message {
        var message: MLXLMCommon.Message = ["role": "assistant"]
        let content = parts.compactMap(\.text).joined()
        let reasoning = parts.compactMap(\.reasoning).joined()
        let toolCalls = parts.compactMap(\.toolCall)
        if !content.isEmpty {
          message["content"] = content
        }
        if !reasoning.isEmpty {
          message["reasoning_content"] = reasoning
          message["thinking"] = reasoning
        }
        if !toolCalls.isEmpty {
          message["tool_calls"] = toolCalls.map(\.mlxToolCall)
        }
        return message
      }

      private static func encode(_ value: EdgeToolsValue) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
      }
    }

    extension EdgeRawToolCall {
      fileprivate var mlxToolCall: MLXLMCommon.Message {
        [
          "type": "function",
          "function": [
            "name": self.name,
            "arguments": self.arguments.mlxValue
          ] as MLXLMCommon.Message
        ]
      }
    }

    extension EdgeToolDefinition {
      public var mlxToolSpec: ToolSpec {
        [
          "type": "function",
          "function": [
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments.edgeToolsValue.mlxValue
          ] as MLXLMCommon.Message
        ]
      }
    }

    extension Sequence where Element == EdgeToolDefinition {
      var mlxToolSpecs: [ToolSpec]? {
        let specifications =
          self
          .filter(\.includesSchemaInInstructions)
          .map(\.mlxToolSpec)
        return specifications.isEmpty ? nil : specifications
      }
    }

    extension EdgeToolsValue {
      fileprivate var mlxValue: any Sendable {
        switch self {
        case .array(let values): values.map(\.mlxValue)
        case .boolean(let value): value
        case .integer(let value): value
        case .null: NSNull()
        case .number(let value): value
        case .object(let object):
          Dictionary(uniqueKeysWithValues: object.map { ($0.key, $0.value.mlxValue) })
        case .string(let value): value
        }
      }
    }

  #endif

  // MARK: - VLM Prompt Conversion

  #if canImport(CoreImage) && canImport(MLXVLM)
    private struct MLXTemporaryVideoInputs: ~Copyable {
      let videos: [UserInput.Video]
      private let temporaryURLs: [URL]

      init<Assets: Sequence>(assets: Assets) throws
      where Assets.Element == EdgeToolsTranscript.Asset {
        var videos = [UserInput.Video]()
        var temporaryURLs = [URL]()
        videos.reserveCapacity(assets.underestimatedCount)

        do {
          for asset in assets {
            switch asset.content {
            case .path(let path):
              videos.append(.url(URL(filePath: path)))
            case .bytes(let bytes):
              let url = Self.temporaryURL(for: asset.mimeType)
              try Data(bytes).write(to: url, options: .atomic)
              videos.append(.url(url))
              temporaryURLs.append(url)
            }
          }
        } catch {
          for url in temporaryURLs {
            try? FileManager.default.removeItem(at: url)
          }
          throw error
        }

        self.videos = videos
        self.temporaryURLs = temporaryURLs
      }

      deinit {
        for url in self.temporaryURLs {
          try? FileManager.default.removeItem(at: url)
        }
      }

      private static func temporaryURL(for mimeType: EdgeToolsMIMEType?) -> URL {
        let fileExtension =
          switch mimeType?.rawValue {
          case EdgeToolsMIMEType.m4v.rawValue: "m4v"
          case EdgeToolsMIMEType.quickTime.rawValue: "mov"
          default: "mp4"
          }
        let directory = FileManager.default.temporaryDirectory
        return directory.appending(path: "EdgeTools-\(UUID().uuidString).\(fileExtension)")
      }
    }

    extension EdgeToolsTranscript.Asset {
      public func mlxImage() throws -> UserInput.Image {
        switch self.content {
        case .path(let path):
          return .url(URL(filePath: path))
        case .bytes(let bytes):
          guard let image = CIImage(data: Data(bytes)) else {
            throw EdgeToolsError.invalidMedia("The image bytes could not be decoded.")
          }
          return .ciImage(image)
        }
      }
    }

    extension Sequence where Element == EdgeToolsTranscript.Asset {
      func mlxImages() throws -> [UserInput.Image] {
        try self.map { try $0.mlxImage() }
      }
    }

    extension EdgeToolsTranscript {
      func rejectAudio() throws {
        guard !self.audio.isEmpty else { return }
        throw EdgeToolsError.unsupportedMedia(
          "This MLX model integration does not support audio input."
        )
      }

      func rejectVideos() throws {
        guard !self.videos.isEmpty else { return }
        throw EdgeToolsError.unsupportedMedia(
          "This MLX model integration does not support video input."
        )
      }

      public func mlxUserInput(
        tools: [EdgeToolDefinition],
        videos: [UserInput.Video] = [],
        additionalContext: [String: any Sendable]? = nil,
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) throws -> UserInput {
        try self.rejectAudio()
        if videos.isEmpty { try self.rejectVideos() }
        return UserInput(
          messages: try self.messages.map(transformMessage),
          images: try self.images.mlxImages(),
          videos: videos,
          tools: tools.mlxToolSpecs,
          additionalContext: additionalContext
        )
      }

      public func mlxVLMInput(
        tools: [EdgeToolDefinition],
        processor: any UserInputProcessor,
        additionalContext: [String: any Sendable]? = nil,
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) async throws -> LMInput {
        let videoInputs = try MLXTemporaryVideoInputs(assets: self.videos)
        return try await processor.prepare(
          input: try self.mlxUserInput(
            tools: tools,
            videos: videoInputs.videos,
            additionalContext: additionalContext
          ) { try transformMessage($0) }
        )
      }
    }
  #endif

  // MARK: - MLXModelState

  public struct MLXModelState<Profile: MLXModelProfile> {
    private struct CachedPrefill {
      let input: LMInput
      let tokenIds: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
      let context: EdgeToolsLLMPrefillContext?
    }

    private struct PrefillCacheState {
      var cachedPrefill: CachedPrefill?
      var inputContext: EdgeToolsLLMPrefillContext?

      mutating func input(for context: EdgeToolsLLMPrefillContext) -> LMInput? {
        self.inputContext = context
        guard let cachedPrefill, cachedPrefill.context == context else { return nil }
        return cachedPrefill.input
      }

      mutating func clearInputContext() {
        self.inputContext = nil
      }

      func matches(input: LMInput) -> Bool {
        mlxPrefillContextMatches(
          cachedInput: self.cachedPrefill?.input,
          input: input,
          cachedContext: self.cachedPrefill?.context,
          inputContext: self.inputContext
        )
      }
    }

    private struct Generation {
      var cache: [any KVCache]
      var outputState: LMOutput.State?
      var logits: MLXArray
      var pendingTokenId: EdgeToolsToken.ID?
      var processor: (any LogitProcessor)?
      let sampler: any LogitSampler
      let synchronizeStreamForMemorySnapshots: Bool
      let generationStartSnapshot: Memory.Snapshot
      let postPrefillSnapshot: Memory.Snapshot
    }

    private let vocabularySizeValue: Int
    private var languageModel: any LanguageModel
    private let processor: (any UserInputProcessor)?
    private let configuredExtraStopTokenIds: Set<EdgeToolsToken.ID>
    private let configuredSampling: EdgeToolsFusedSamplingParameters?
    private var prefillCacheState = PrefillCacheState()
    private var generation: Generation?

    public init(
      languageModel: any LanguageModel,
      processor: (any UserInputProcessor)? = nil,
      vocabularySize: Int,
      extraStopTokenIds: Set<EdgeToolsToken.ID>,
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) {
      self.languageModel = languageModel
      self.processor = processor
      self.vocabularySizeValue = vocabularySize
      self.configuredExtraStopTokenIds = extraStopTokenIds
      self.configuredSampling = defaultSampling
    }

    public var vocabularySize: Int { self.vocabularySizeValue }
    public var extraStopTokenIds: Set<EdgeToolsToken.ID> {
      self.configuredExtraStopTokenIds
    }

    public func grammarEngine(
      tokenizer: any EdgeToolsTokenizer
    ) throws -> Profile.GrammarEngine {
      var stopTokenIds = self.extraStopTokenIds
      if let eosTokenId = tokenizer.eosTokenId { stopTokenIds.insert(eosTokenId) }
      return try Profile.grammarEngine(
        tokenizer: tokenizer,
        vocabularySize: self.vocabularySize,
        stopTokenIds: stopTokenIds
      )
    }

    public func grammar(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      grammarEngine: borrowing Profile.GrammarEngine
    ) throws -> Profile.GrammarEngine.Grammar {
      try Profile.grammar(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      )
    }

    public nonisolated(nonsending) mutating func prepare(
      prompt: inout Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser
    ) async throws -> EdgeToolsModelPreparation {
      Profile.prepare(prompt: &prompt, tools: tools, parser: &parser)
      let input = try await self.input(prompt: prompt, tools: tools, tokenizer: tokenizer)
      let defaultSampling =
        Profile.defaultSampling(prompt: prompt, parameters: parameters)
        ?? self.configuredSampling
        ?? EdgeToolsFusedSamplingParameters()
      let sampler =
        parameters.sampler
        ?? MLXFusedSampler(parameters: parameters.samplingOverrides.applying(to: defaultSampling))
      return try await self.prepare(input: input, sampler: sampler, parameters: parameters)
    }

    public nonisolated(nonsending) mutating func tokenIds(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> [EdgeToolsToken.ID] {
      let input = try await self.input(prompt: prompt, tools: tools, tokenizer: tokenizer)
      return input.text.tokens.asArray(EdgeToolsToken.ID.self)
    }

    private nonisolated(nonsending) mutating func prepare(
      input: LMInput,
      sampler: any LogitSampler,
      parameters: Profile.GenerateParameters
    ) async throws -> EdgeToolsModelPreparation {
      let clock = ContinuousClock()
      let generationStartSnapshot = Self.memorySnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let tokenIds = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let start = clock.now
      var processor = parameters.processor
      processor?.prompt(input.text.tokens)
      let prepared = try self.preparedOutput(input: input, tokenIds: tokenIds)
      let metrics = EdgeToolsPrefillMetrics(
        tokens: prepared.tokenCount,
        duration: start.duration(to: clock.now)
      )
      let postPrefillSnapshot = Self.memorySnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      self.generation = Generation(
        cache: prepared.cache,
        outputState: prepared.output.state,
        logits: prepared.output.logits,
        pendingTokenId: nil,
        processor: processor,
        sampler: sampler,
        synchronizeStreamForMemorySnapshots: parameters.synchronizeStreamForMemorySnapshots,
        generationStartSnapshot: generationStartSnapshot,
        postPrefillSnapshot: postPrefillSnapshot
      )
      return EdgeToolsModelPreparation(metrics: metrics)
    }

    public nonisolated(nonsending) mutating func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters
    ) async throws -> EdgeToolsModelSample {
      guard var generation = self.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      if let pendingTokenId = generation.pendingTokenId {
        maybeQuantizeKVCache(
          cache: &generation.cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize,
          quantizedKVStart: parameters.quantizedKVStart
        )
        let token = MLXArray([pendingTokenId])
        let output = self.languageModel(
          LMInput.Text(tokens: token)[text: .newAxis],
          cache: generation.cache,
          state: generation.outputState
        )
        generation.outputState = output.state
        generation.logits = output.logits
      }
      var stepLogits = generation.logits[0..., -1, 0...]
      stepLogits = generation.processor?.process(logits: stepLogits) ?? stepLogits
      let maskedLogits = applyBitmaskMLX(logits: stepLogits, mask: bitmask)
      let confidenceValues = top(maskedLogits.flattened(), k: 2)
      let token = generation.sampler.sample(logits: maskedLogits)
      eval(confidenceValues, token)

      let confidence = tokenConfidence(unorderedPair: confidenceValues.asArray(Float.self))
      let tokenId = token.item(EdgeToolsToken.ID.self)
      generation.processor?.didSample(token: token)
      generation.pendingTokenId = tokenId
      self.generation = generation
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public func finish() -> EdgeToolsMetadata {
      guard let generation = self.generation else { return EdgeToolsMetadata() }
      var metadata = EdgeToolsMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = generation.generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = generation.postPrefillSnapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = Self.memorySnapshot(
        synchronize: generation.synchronizeStreamForMemorySnapshots
      )
      return metadata
    }

    public mutating func resetGeneration() async {
      self.generation = nil
    }

    public func contextState() -> sending Self {
      // Model weights are immutable after engine initialization and intentionally shared across
      // contexts. All mutable generation and prefill state is created afresh below.
      nonisolated(unsafe) let languageModel = self.languageModel
      return Self(
        languageModel: languageModel,
        processor: self.processor,
        vocabularySize: self.vocabularySizeValue,
        extraStopTokenIds: self.configuredExtraStopTokenIds,
        defaultSampling: self.configuredSampling
      )
    }

    public nonisolated(nonsending) mutating func prefill(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsEnginePrefill {
      let input = try await self.input(prompt: prompt, tools: tools, tokenizer: tokenizer)
      return try await self.prefill(input: input)
    }

    private nonisolated(nonsending) mutating func prefill(
      input: LMInput
    ) async throws -> EdgeToolsEnginePrefill {
      let clock = ContinuousClock()
      let tokenIds = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let start = clock.now
      let prepared = try self.preparedOutput(input: input, tokenIds: tokenIds)
      eval(prepared.output.logits)
      eval(prepared.cache)
      self.prefillCacheState.cachedPrefill = CachedPrefill(
        input: input,
        tokenIds: tokenIds,
        cache: prepared.cache.map { $0.copy() },
        output: prepared.output,
        context: self.prefillCacheState.inputContext
      )
      let snapshot = Self.memorySnapshot(synchronize: true)
      var metadata = EdgeToolsMetadata()
      metadata.mlxEnginePostPrefillMemorySnapshot = snapshot
      return EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(
          tokens: prepared.tokenCount,
          duration: start.duration(to: clock.now)
        ),
        metadata: metadata
      )
    }

    private nonisolated(nonsending) mutating func input(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> LMInput {
      if let prompt = prompt as? EdgeToolsTranscript {
        let context = EdgeToolsLLMPrefillContext(prompt: prompt, tools: tools)
        if let input = self.prefillCacheState.input(for: context) {
          return input
        }
      } else {
        self.prefillCacheState.clearInputContext()
      }
      return try await Profile.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        processor: self.processor
      )
    }

    private func preparedOutput(
      input: LMInput,
      tokenIds: [EdgeToolsToken.ID]
    ) throws -> (output: LMOutput, cache: [any KVCache], tokenCount: Int) {
      guard let cachedPrefill = self.prefillCacheState.cachedPrefill,
        tokenIds.starts(with: cachedPrefill.tokenIds),
        self.prefillCacheState.matches(input: input)
      else {
        let cache = self.languageModel.newCache(parameters: nil)
        return (try self.prepareModelOutput(input: input, cache: cache), cache, tokenIds.count)
      }
      let suffixCount = tokenIds.count - cachedPrefill.tokenIds.count
      let cache = cachedPrefill.cache.map { $0.copy() }
      let output =
        if suffixCount == 0 {
          cachedPrefill.output
        } else {
          self.languageModel(
            mlxTextSuffix(input.text, from: cachedPrefill.tokenIds.count),
            cache: cache,
            state: cachedPrefill.output.state
          )
        }
      return (output, cache, suffixCount)
    }

    private func prepareModelOutput(input: LMInput, cache: [any KVCache]) throws -> LMOutput {
      switch try self.languageModel.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw MLXEngineError(code: .emptyInput, message: "Model received empty input.")
        }
        return self.languageModel(
          tokens[text: .newAxis],
          cache: cache.isEmpty ? nil : cache,
          state: nil
        )
      }
    }

    private static func memorySnapshot(synchronize: Bool) -> Memory.Snapshot {
      if synchronize {
        Stream.defaultStream(.defaultDevice()).synchronize()
      }
      return Memory.snapshot()
    }
  }

  // MARK: - MLXGenerationState

  public struct MLXGenerationState<Profile: MLXModelProfile>
  where Profile.Prompt == EdgeToolsTranscript {
    public var model: MLXModelState<Profile>
    public var transcript: EdgeToolsTranscript
    public let revision: Int

    public init(
      model: MLXModelState<Profile>,
      transcript: EdgeToolsTranscript,
      revision: Int
    ) {
      self.model = model
      self.transcript = transcript
      self.revision = revision
    }
  }

  extension MLXEngine {
    public convenience init(
      from directoryURL: URL,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void = { _, _ in }
    ) async throws where Profile: MLXLLMModelProfile {
      try await self.init(
        from: MLXModelDirectory(url: directoryURL),
        patchWeights: patchWeights
      )
    }

    public convenience init(
      from directory: MLXModelDirectory,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void = { _, _ in }
    ) async throws where Profile: MLXLLMModelProfile {
      let configurationData = try directory.loadConfigurationData()
      let baseConfiguration = try JSONDecoder.json5()
        .decode(BaseConfiguration.self, from: configurationData)
      let tokenizer = try await directory.loadTokenizer()
      let languageModel = try await LLMTypeRegistry.shared.createModel(
        configuration: configurationData,
        modelType: baseConfiguration.modelType
      )
      try self.init(
        from: directory,
        configurationData: configurationData,
        baseConfiguration: baseConfiguration,
        languageModel: languageModel,
        tokenizer: tokenizer,
        processor: nil,
        patchWeights: patchWeights
      )
    }

    #if canImport(CoreImage) && canImport(MLXVLM) && canImport(Tokenizers)
      public convenience init(
        from directoryURL: URL,
        patchWeights: (
          _ weights: inout [String: MLXArray],
          _ model: any LanguageModel
        ) throws -> Void = { _, _ in }
      ) async throws where Profile: MLXVLMModelProfile {
        try await self.init(
          from: MLXModelDirectory(url: directoryURL),
          patchWeights: patchWeights
        )
      }

      public convenience init(
        from directory: MLXModelDirectory,
        patchWeights: (
          _ weights: inout [String: MLXArray],
          _ model: any LanguageModel
        ) throws -> Void = { _, _ in }
      ) async throws where Profile: MLXVLMModelProfile {
        let configurationData = try directory.loadConfigurationData()
        let baseConfiguration = try JSONDecoder.json5()
          .decode(BaseConfiguration.self, from: configurationData)
        let tokenizer = try await directory.loadTokenizer()
        guard let tokenizer = tokenizer as? TransformersTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let languageModel = try await VLMTypeRegistry.shared.createModel(
          configuration: configurationData,
          modelType: baseConfiguration.modelType
        )
        let processor = try await mlxVLMProcessor(
          from: directory,
          modelType: baseConfiguration.modelType,
          tokenizer: tokenizer
        )
        try self.init(
          from: directory,
          configurationData: configurationData,
          baseConfiguration: baseConfiguration,
          languageModel: languageModel,
          tokenizer: tokenizer,
          processor: processor,
          patchWeights: patchWeights
        )
      }
    #endif

    private convenience init(
      from directory: MLXModelDirectory,
      configurationData: Data,
      baseConfiguration: BaseConfiguration,
      languageModel: sending any LanguageModel,
      tokenizer: sending any EdgeToolsTokenizer,
      processor: sending (any UserInputProcessor)?,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void
    ) throws {
      let loadedLanguageModel = try loadMLXWeights(
        from: directory,
        into: languageModel,
        configuration: baseConfiguration,
        patchWeights: patchWeights
      )
      let vocabularySize = try mlxVocabularySize(from: configurationData)
      let extraStopTokenIds = try mlxExtraStopTokenIds(
        profile: Profile.self,
        directory: directory,
        tokenizer: tokenizer
      )
      try self.init(
        languageModel: loadedLanguageModel,
        tokenizer: tokenizer,
        processor: processor,
        vocabularySize: vocabularySize,
        extraStopTokenIds: extraStopTokenIds,
        defaultSampling: try? directory.loadDefaultSampling()
      )
    }
  }

  // MARK: - MLXEngine

  public final class MLXEngine<Profile: MLXModelProfile>:
    EdgeToolsModelEngine, EdgeToolsPrefillableEngine
  where Profile.Prompt == EdgeToolsTranscript {
    public typealias Context = MLXContext<Profile>
    public typealias ContextParameters = MLXContextParameters
    public typealias Prompt = EdgeToolsTranscript.UserMessage
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias ModelGenerationState = MLXGenerationState<Profile>
    public typealias GenerationParser = Profile.GenerationParser
    public typealias GrammarEngine = Profile.GrammarEngine

    private let prototype: Lock<MLXModelState<Profile>>
    private let extraStopTokenIds: Set<EdgeToolsToken.ID>
    public let tokenizer: any EdgeToolsTokenizer
    public let grammarEngine: Profile.GrammarEngine

    public init(
      languageModel: sending any LanguageModel,
      tokenizer: sending any EdgeToolsTokenizer,
      processor: sending (any UserInputProcessor)? = nil,
      vocabularySize: Int,
      extraStopTokenIds: Set<EdgeToolsToken.ID> = [],
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      let prototype = MLXModelState<Profile>(
        languageModel: languageModel,
        processor: processor,
        vocabularySize: vocabularySize,
        extraStopTokenIds: extraStopTokenIds,
        defaultSampling: defaultSampling
      )
      self.grammarEngine = try prototype.grammarEngine(tokenizer: tokenizer)
      self.prototype = Lock(prototype)
      self.extraStopTokenIds = extraStopTokenIds
      self.tokenizer = tokenizer
    }

    public func context() -> MLXContext<Profile> {
      self.context(MLXContextParameters())
    }

    public func context(_ parameters: MLXContextParameters) -> MLXContext<Profile> {
      MLXContext(
        parameters: parameters,
        model: self.makeModelState(),
        makeModelState: { self.makeModelState() }
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> [EdgeToolsToken] {
      let transcript = context.transcript(appending: prompt)
      var model = self.makeModelState()
      let tokenIds = try await model.tokenIds(
        prompt: transcript,
        tools: tools,
        tokenizer: self.tokenizer
      )
      let tokens = self.tokenizer.convertIdsToTokens(tokenIds)
      return zip(tokenIds, tokens)
        .compactMap { tokenId, token in
          token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
        }
    }

    public func generationState(
      prompt: EdgeToolsTranscript.UserMessage,
      context: MLXContext<Profile>
    ) async throws -> ModelGenerationState {
      self.generationState(from: try context.begin(appending: prompt))
    }

    public func prepare(
      prompt: inout EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsModelPreparation {
      try await state.model.prepare(
        prompt: &state.transcript,
        tools: tools,
        tokenizer: self.tokenizer,
        parameters: parameters,
        parser: &parser
      )
    }

    public func grammar(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      state: ModelGenerationState
    ) throws -> Profile.GrammarEngine.Grammar {
      try state.model.grammar(
        prompt: state.transcript,
        tools: tools,
        parameters: parameters,
        grammarEngine: self.grammarEngine
      )
    }

    public func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsModelSample {
      try await state.model.decode(bitmask: bitmask, parameters: parameters)
    }

    public func stopTokenIds(state: ModelGenerationState) -> Set<EdgeToolsToken.ID> {
      var stopTokenIds = self.extraStopTokenIds
      if let eosTokenId = self.tokenizer.eosTokenId {
        stopTokenIds.insert(eosTokenId)
      }
      return stopTokenIds
    }

    public func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: MLXContext<Profile>
    ) async -> Result<EdgeToolsEngineGeneration, any Error> {
      var state = state
      let metadata = state.model.finish()
      await state.model.resetGeneration()

      let generation: EdgeToolsEngineGeneration?
      let finalResult: Result<EdgeToolsEngineGeneration, any Error>
      switch result {
      case .success(var value):
        value.metadata.merge(metadata) { _, finalValue in finalValue }
        generation = value
        finalResult = .success(value)
      case .failure(let error):
        generation = nil
        finalResult = .failure(error)
      }
      // The state is returned exactly once and is not accessed after this point. Region isolation
      // does not currently infer that exclusivity through the synchronous context method.
      nonisolated(unsafe) let restoredModel = state.model
      context.finish(
        generation: generation,
        revision: state.revision,
        model: restoredModel
      )
      return finalResult
    }

    public func prefill(
      promptPrefix: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try await self.prefill(
        snapshot: context.begin(appending: promptPrefix),
        tools: tools,
        context: context
      )
    }

    public func prefill(
      tools: [EdgeToolDefinition] = [],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try await self.prefill(
        snapshot: context.begin(),
        tools: tools,
        context: context
      )
    }

    public func generate(
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      self.generationTask(
        prompt: .user(""),
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          self.generationState(from: try context.begin())
        }
      )
    }

    private func prefill(
      snapshot: MLXContext<Profile>.GenerationSnapshot,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      var model = snapshot.model
      do {
        let prefill = try await model.prefill(
          prompt: snapshot.transcript,
          tools: tools,
          tokenizer: self.tokenizer
        )
        // The model is no longer used after it is returned to the context. Region isolation does
        // not currently infer that exclusivity through this synchronous transfer.
        nonisolated(unsafe) let restoredModel = model
        context.finish(generation: nil, revision: snapshot.revision, model: restoredModel)
        return prefill
      } catch {
        await model.resetGeneration()
        nonisolated(unsafe) let restoredModel = model
        context.finish(generation: nil, revision: snapshot.revision, model: restoredModel)
        throw error
      }
    }

    private func makeModelState() -> sending MLXModelState<Profile> {
      self.prototype.withBorrowedLock { $0.contextState() }
    }

    private func generationState(
      from snapshot: MLXContext<Profile>.GenerationSnapshot
    ) -> ModelGenerationState {
      ModelGenerationState(
        model: snapshot.model,
        transcript: snapshot.transcript,
        revision: snapshot.revision
      )
    }
  }

  #if XGrammar
    extension MLXEngine where Profile.GrammarEngine == XGrammarEngine {
      public func clearCaches() {
        self.grammarEngine.clearCaches()
      }
    }

    extension EdgeToolsSession {
      public func clearCaches<Profile>()
        where Engine == MLXEngine<Profile>, Profile.GrammarEngine == XGrammarEngine
      {
        self.engine.clearCaches()
      }
    }
  #endif

  extension EdgeToolsSession {
    public func context<Profile>(
      transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> MLXContext<Profile> where Engine == MLXEngine<Profile> {
      self.context(MLXContextParameters(transcript: transcript, reasoningEffort: reasoningEffort))
    }

    public func context<Profile>(
      systemPrompt: String,
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> MLXContext<Profile> where Engine == MLXEngine<Profile> {
      let transcript = EdgeToolsTranscript(messages: [.system(systemPrompt)])
      return self.context(transcript: transcript, reasoningEffort: reasoningEffort)
    }
  }

  private struct MLXVocabularyConfiguration: Decodable {
    struct TextConfiguration: Decodable {
      var vocabularySize: Int

      enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
      }
    }

    var vocabularySize: Int?
    var textConfiguration: TextConfiguration?

    enum CodingKeys: String, CodingKey {
      case vocabularySize = "vocab_size"
      case textConfiguration = "text_config"
    }
  }

  private func mlxVocabularySize(from configurationData: Data) throws -> Int {
    let configuration = try JSONDecoder.json5()
      .decode(
        MLXVocabularyConfiguration.self,
        from: configurationData
      )
    guard
      let vocabularySize = configuration.vocabularySize
        ?? configuration.textConfiguration?.vocabularySize
    else {
      throw EdgeToolsError.failedToLoadConfiguration
    }
    return vocabularySize
  }

  private func mlxExtraStopTokenIds<Profile: MLXModelProfile>(
    profile: Profile.Type,
    directory: MLXModelDirectory,
    tokenizer: any EdgeToolsTokenizer
  ) throws -> Set<EdgeToolsToken.ID> {
    var tokenIds = try directory.loadStopTokenIds()
    tokenIds.formUnion(Profile.extraStopTokens.compactMap { tokenizer.convertTokenToId($0) })
    if let eosTokenId = tokenizer.eosTokenId { tokenIds.remove(eosTokenId) }
    return tokenIds
  }

  #if canImport(CoreImage) && canImport(MLXVLM) && canImport(Tokenizers)
    private func mlxVLMProcessor(
      from directory: MLXModelDirectory,
      modelType: String,
      tokenizer: TransformersTokenizer
    ) async throws -> sending any UserInputProcessor {
      let data = try directory.loadProcessorConfigurationData()
      let configuration = try JSONDecoder.json5()
        .decode(
          BaseProcessorConfiguration.self,
          from: data
        )
      let processorType =
        switch modelType {
        case "mistral3": "Mistral3Processor"
        case "gemma4_unified": "Gemma4UnifiedProcessor"
        default: configuration.processorClass
        }
      return try await VLMProcessorTypeRegistry.shared.createModel(
        configuration: data,
        processorType: processorType,
        tokenizer: adaptedMLXTokenizer(tokenizer.base)
      )
    }

    private func adaptedMLXTokenizer(
      _ tokenizer: any Tokenizers.Tokenizer
    ) -> any MLXLMCommon.Tokenizer {
      #adaptHuggingFaceTokenizer(tokenizer)
    }
  #endif

  private func loadMLXWeights(
    from directory: MLXModelDirectory,
    into model: sending any LanguageModel,
    configuration: BaseConfiguration,
    patchWeights: (
      _ weights: inout [String: MLXArray],
      _ model: any LanguageModel
    ) throws -> Void
  ) throws -> sending any LanguageModel {
    let safetensors = try directory.loadSafetensors()
    var weights = model.sanitize(
      weights: safetensors.weights,
      metadata: safetensors.mergedMetadata
    )
    try patchWeights(&weights, model)
    if let perLayerQuantization = configuration.perLayerQuantization {
      quantize(model: model) { path, _ in
        guard weights["\(path).scales"] != nil else { return nil }
        return perLayerQuantization.quantization(layer: path)?.asTuple
      }
    }
    try model.update(
      parameters: ModuleParameters.unflattened(weights),
      verify: [.all]
    )
    eval(model)
    return model
  }

  private func mlxPrefillContextMatches(
    cachedInput: LMInput?,
    input: LMInput,
    cachedContext: EdgeToolsLLMPrefillContext?,
    inputContext: EdgeToolsLLMPrefillContext?
  ) -> Bool {
    guard let cachedInput,
      mlxTextMasksHaveSamePrefix(cachedInput.text.mask, input.text.mask)
    else {
      return false
    }
    if let cachedContext, let inputContext {
      return cachedContext.hasMediaPrefix(in: inputContext)
    }
    return mlxProcessedImagesEqual(cachedInput.image, input.image)
      && mlxProcessedVideosEqual(cachedInput.video, input.video)
      && mlxProcessedAudioEqual(cachedInput.audio, input.audio)
  }

  private func mlxTextSuffix(_ text: LMInput.Text, from index: Int) -> LMInput.Text {
    let tokens =
      if text.tokens.ndim == 1 {
        text.tokens[index...][.newAxis]
      } else {
        text.tokens[.ellipsis, index...]
      }
    let mask = text.mask.map { mask in
      if mask.ndim == 1 {
        mask[index...][.newAxis]
      } else {
        mask[.ellipsis, index...]
      }
    }
    return LMInput.Text(tokens: tokens, mask: mask)
  }

  private func mlxTextMasksHaveSamePrefix(_ lhs: MLXArray?, _ rhs: MLXArray?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      return true
    case (.some(let lhs), .some(let rhs)):
      guard lhs.ndim == rhs.ndim, lhs.shape.dropLast() == rhs.shape.dropLast(),
        lhs.dim(-1) <= rhs.dim(-1)
      else { return false }
      return mlxArraysEqual(lhs, rhs[.ellipsis, ..<lhs.dim(-1)])
    default:
      return false
    }
  }

  private func mlxProcessedImagesEqual(
    _ lhs: LMInput.ProcessedImage?,
    _ rhs: LMInput.ProcessedImage?
  ) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      mlxArraysEqual(lhs.pixels, rhs.pixels)
        && mlxArraysEqual(lhs.positionIds, rhs.positionIds)
        && mlxFramesEqual(lhs.frames, rhs.frames)
    default:
      false
    }
  }

  private func mlxProcessedVideosEqual(
    _ lhs: LMInput.ProcessedVideo?,
    _ rhs: LMInput.ProcessedVideo?
  ) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      mlxArraysEqual(lhs.pixels, rhs.pixels)
        && mlxArraysEqual(lhs.positionIds, rhs.positionIds)
        && mlxFramesEqual(lhs.frames, rhs.frames)
    default:
      false
    }
  }

  private func mlxProcessedAudioEqual(
    _ lhs: LMInput.ProcessedAudio?,
    _ rhs: LMInput.ProcessedAudio?
  ) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      mlxArraysEqual(lhs.features, rhs.features) && mlxArraysEqual(lhs.mask, rhs.mask)
    default:
      false
    }
  }

  private func mlxArraysEqual(_ lhs: MLXArray?, _ rhs: MLXArray?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      lhs.dtype == rhs.dtype && lhs.shape == rhs.shape && lhs.arrayEqual(rhs).item(Bool.self)
    default:
      false
    }
  }

  private func mlxFramesEqual(_ lhs: [THW]?, _ rhs: [THW]?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      lhs.count == rhs.count
        && zip(lhs, rhs)
          .allSatisfy { lhs, rhs in lhs.t == rhs.t && lhs.h == rhs.h && lhs.w == rhs.w }
    default:
      false
    }
  }
#endif
