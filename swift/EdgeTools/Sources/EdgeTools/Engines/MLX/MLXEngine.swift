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

    static nonisolated(nonsending) func prefillInput(
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

    public static nonisolated(nonsending) func prefillInput(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        processor: processor
      )
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

  fileprivate final class MLXEngineIdentity: Sendable {}

  public final class MLXContext<Profile: MLXModelProfile>: Identifiable, Sendable
  where Profile.Prompt == EdgeToolsTranscript {
    private struct State {
      var transcript: EdgeToolsTranscript
      var reasoningEffort: EdgeToolsReasoningEffort
      var isResponding = false
      var revision = 0
      var model: MLXModelState<Profile>

      mutating func finish(
        generation: EdgeToolsEngineGeneration?,
        revision: Int,
        model: sending MLXModelState<Profile>
      ) {
        if self.revision == revision {
          self.model = model
        }
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

    private struct ForkSnapshot {
      let parameters: MLXContextParameters
      let model: MLXModelState<Profile>
    }

    private let state: Lock<State>
    fileprivate let engineIdentity: MLXEngineIdentity
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

    fileprivate init(
      parameters: MLXContextParameters,
      model: sending MLXModelState<Profile>,
      engineIdentity: MLXEngineIdentity
    ) {
      self.state = Lock(
        State(
          transcript: parameters.transcript,
          reasoningEffort: parameters.reasoningEffort,
          model: model
        )
      )
      self.engineIdentity = engineIdentity
    }

    public func fork() -> MLXContext<Profile> {
      let snapshot = self.state.withBorrowedLock { state in
        ForkSnapshot(
          parameters: MLXContextParameters(
            transcript: state.transcript,
            reasoningEffort: state.reasoningEffort
          ),
          model: state.model.forkedContextState(copyingCache: state.isResponding)
        )
      }
      return MLXContext(
        parameters: snapshot.parameters,
        model: snapshot.model,
        engineIdentity: self.engineIdentity
      )
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
        guard !state.isResponding else {
          throw EdgeToolsError.contextInUse
        }
        let model = state.model.forkedContextState(copyingCache: false)
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
        guard !state.isResponding else {
          throw EdgeToolsError.contextInUse
        }
        let model = state.model.forkedContextState(copyingCache: false)
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
      // The compiler cannot track a `sending` parameter through the lock closure. This method
      // stores the generation branch once when its context revision still matches.
      nonisolated(unsafe) let model = model
      self.state.withLock { state in
        state.finish(
          generation: generation,
          revision: revision,
          model: model
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
        try self.input(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          addGenerationPrompt: true
        )
      }

      public static nonisolated(nonsending) func prefillInput(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        processor: (any UserInputProcessor)?
      ) async throws -> LMInput {
        try self.input(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          addGenerationPrompt: false
        )
      }

      private static func input(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        addGenerationPrompt: Bool
      ) throws -> LMInput {
        guard let tokenizer = tokenizer as? TransformersTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let tokenIds = try tokenizer.base.applyChatTemplate(
          messages: try prompt.mlxMessages(),
          chatTemplate: nil,
          addGenerationPrompt: addGenerationPrompt,
          truncation: false,
          maxLength: nil,
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

  private enum MLXInputKind: Equatable {
    case generation
    case prefill
  }

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
    private final class CachedPrefill {
      struct Snapshot {
        let input: LMInput
        let tokenIds: [EdgeToolsToken.ID]
        let cache: [any KVCache]
        let output: LMOutput
        let context: EdgeToolsLLMPrefillContext?
      }

      let input: LMInput
      let tokenIds: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
      let context: EdgeToolsLLMPrefillContext?
      let inputKind: MLXInputKind
      private let copying = Lock(())

      init(
        input: LMInput,
        tokenIds: [EdgeToolsToken.ID],
        cache: [any KVCache],
        output: LMOutput,
        context: EdgeToolsLLMPrefillContext?,
        inputKind: MLXInputKind
      ) {
        self.input = input
        self.tokenIds = tokenIds
        self.cache = cache
        self.output = output
        self.context = context
        self.inputKind = inputKind
      }

      func input(
        for context: EdgeToolsLLMPrefillContext,
        kind: MLXInputKind
      ) -> LMInput? {
        self.context == context && self.inputKind == kind ? self.input : nil
      }

      func mutableSnapshot(
        tokenIds: [EdgeToolsToken.ID],
        input: LMInput,
        inputContext: EdgeToolsLLMPrefillContext?
      ) -> Snapshot? {
        guard tokenIds.starts(with: self.tokenIds),
          mlxPrefillContextMatches(
            cachedInput: self.input,
            input: input,
            cachedContext: self.context,
            inputContext: inputContext
          )
        else {
          return nil
        }
        return Snapshot(
          input: self.input,
          tokenIds: self.tokenIds,
          cache: self.copiedCache(),
          output: self.output,
          context: self.context
        )
      }

      func forked(copyingCache: Bool) -> CachedPrefill {
        guard copyingCache else {
          return self
        }
        let cache = self.copiedCache()
        eval(cache)
        return CachedPrefill(
          input: self.input,
          tokenIds: self.tokenIds,
          cache: cache,
          output: self.output,
          context: self.context,
          inputKind: self.inputKind
        )
      }

      private func copiedCache() -> [any KVCache] {
        // The lock closure runs synchronously and exclusively. The local only escapes after the
        // copy completes, but Swift's isolation checker cannot express that transfer today.
        nonisolated(unsafe) var copiedCache: [any KVCache]?
        self.copying.withLock { _ in
          copiedCache = self.cache.map { $0.copy() }
        }
        return copiedCache!
      }
    }

    private struct PrefillCacheState {
      var cachedPrefill: CachedPrefill?
      var inputContext: EdgeToolsLLMPrefillContext?

      mutating func input(
        for context: EdgeToolsLLMPrefillContext,
        kind: MLXInputKind
      ) -> LMInput? {
        self.inputContext = context
        return self.cachedPrefill?.input(for: context, kind: kind)
      }

      mutating func clearInputContext() {
        self.inputContext = nil
      }

      mutating func fork(copyingCache: Bool) {
        self.cachedPrefill = self.cachedPrefill?.forked(copyingCache: copyingCache)
      }
    }

    private struct Generation {
      let input: LMInput
      var cachedTokenIds: [EdgeToolsToken.ID]
      var cache: [any KVCache]
      var outputState: LMOutput.State?
      var logits: MLXArray
      var pendingTokenId: EdgeToolsToken.ID?
      let inputContext: EdgeToolsLLMPrefillContext?
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
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
      Profile.prepare(prompt: &prompt, tools: tools, parser: &parser)
      let input = try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        kind: .generation
      )
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
      let input = try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        kind: .generation
      )
      return input.text.tokens.asArray(EdgeToolsToken.ID.self)
    }

    private nonisolated(nonsending) mutating func prepare(
      input: LMInput,
      sampler: any LogitSampler,
      parameters: Profile.GenerateParameters
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
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
        input: input,
        cachedTokenIds: tokenIds,
        cache: prepared.cache,
        outputState: prepared.output.state,
        logits: prepared.output.logits,
        pendingTokenId: nil,
        inputContext: self.prefillCacheState.inputContext,
        processor: processor,
        sampler: sampler,
        synchronizeStreamForMemorySnapshots: parameters.synchronizeStreamForMemorySnapshots,
        generationStartSnapshot: generationStartSnapshot,
        postPrefillSnapshot: postPrefillSnapshot
      )
      return EdgeToolsGenerationLoop.Preparation(metrics: metrics)
    }

    public nonisolated(nonsending) mutating func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters
    ) async throws -> EdgeToolsGenerationLoop.Sample {
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
        generation.cachedTokenIds.append(pendingTokenId)
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
      return EdgeToolsGenerationLoop.Sample(tokenId: tokenId, confidence: confidence)
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

    public mutating func commitGeneration(
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) {
      guard var generation = self.generation else { return }
      if let pendingTokenId = generation.pendingTokenId,
        !stopTokenIds.contains(pendingTokenId)
      {
        let output = self.languageModel(
          LMInput.Text(tokens: MLXArray([pendingTokenId]))[text: .newAxis],
          cache: generation.cache,
          state: generation.outputState
        )
        generation.outputState = output.state
        generation.logits = output.logits
        generation.cachedTokenIds.append(pendingTokenId)
      }
      eval(generation.logits)
      eval(generation.cache)
      self.prefillCacheState.cachedPrefill = CachedPrefill(
        input: generation.input,
        tokenIds: generation.cachedTokenIds,
        cache: generation.cache,
        output: LMOutput(logits: generation.logits, state: generation.outputState),
        context: generation.inputContext,
        inputKind: .generation
      )
      self.generation = nil
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

    fileprivate func forkedContextState(copyingCache: Bool) -> sending Self {
      // Model weights are immutable after engine initialization and intentionally shared across
      // contexts. The cached prefill is immutable, or is copied here for an active-context fork.
      nonisolated(unsafe) let languageModel = self.languageModel
      var state = Self(
        languageModel: languageModel,
        processor: self.processor,
        vocabularySize: self.vocabularySizeValue,
        extraStopTokenIds: self.configuredExtraStopTokenIds,
        defaultSampling: self.configuredSampling
      )
      state.prefillCacheState = self.prefillCacheState
      state.prefillCacheState.fork(copyingCache: copyingCache)
      // The returned state shares only immutable model weights and, for an idle context, an
      // immutable cached-prefill checkpoint whose cache copies are serialized internally. Swift's
      // isolation checker cannot express that ownership boundary.
      nonisolated(unsafe) let transferredState = state
      return transferredState
    }

    public nonisolated(nonsending) mutating func prefill(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsEnginePrefill {
      let input = try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        kind: .prefill
      )
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
        cache: prepared.cache,
        output: prepared.output,
        context: self.prefillCacheState.inputContext,
        inputKind: .prefill
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
      tokenizer: any EdgeToolsTokenizer,
      kind: MLXInputKind
    ) async throws -> LMInput {
      if let prompt = prompt as? EdgeToolsTranscript {
        let context = EdgeToolsLLMPrefillContext(prompt: prompt, tools: tools)
        if let input = self.prefillCacheState.input(for: context, kind: kind) {
          return input
        }
      } else {
        self.prefillCacheState.clearInputContext()
      }
      switch kind {
      case .generation:
        return try await Profile.input(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          processor: self.processor
        )
      case .prefill:
        return try await Profile.prefillInput(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          processor: self.processor
        )
      }
    }

    private func preparedOutput(
      input: LMInput,
      tokenIds: [EdgeToolsToken.ID]
    ) throws -> (output: LMOutput, cache: [any KVCache], tokenCount: Int) {
      guard
        let cachedPrefill = self.prefillCacheState.cachedPrefill?.mutableSnapshot(
          tokenIds: tokenIds,
          input: input,
          inputContext: self.prefillCacheState.inputContext
        )
      else {
        let cache = self.languageModel.newCache(parameters: nil)
        return (try self.prepareModelOutput(input: input, cache: cache), cache, tokenIds.count)
      }
      let suffixCount = tokenIds.count - cachedPrefill.tokenIds.count
      let cache = cachedPrefill.cache
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
    EdgeToolsEngine, EdgeToolsPrefillableEngine
  where Profile.Prompt == EdgeToolsTranscript {
    public typealias Context = MLXContext<Profile>
    public typealias ContextParameters = MLXContextParameters
    public typealias Prompt = EdgeToolsTranscript.UserMessage
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias ModelGenerationState = MLXGenerationState<Profile>
    public typealias GenerationParser = Profile.GenerationParser
    public typealias GrammarEngine = Profile.GrammarEngine

    private let prototype: Lock<MLXModelState<Profile>>
    private let identity = MLXEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
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
      self.generationLoop = EdgeToolsGenerationLoop(
        tokenizer: tokenizer,
        extraStopTokenIds: extraStopTokenIds
      )
      self.tokenizer = tokenizer
    }

    public func context() -> MLXContext<Profile> {
      self.context(MLXContextParameters())
    }

    public func context(_ parameters: MLXContextParameters) -> MLXContext<Profile> {
      MLXContext(
        parameters: parameters,
        model: self.makeModelState(),
        engineIdentity: self.identity
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> [EdgeToolsToken] {
      try self.validate(context)
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
      try self.validate(context)
      return self.generationState(from: try context.begin(appending: prompt))
    }

    public func generate(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      self.generationTask(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          try await self.generationState(prompt: prompt, context: context)
        }
      )
    }

    public func prepare(
      prompt: inout EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
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
    ) async throws -> EdgeToolsGenerationLoop.Sample {
      try await state.model.decode(bitmask: bitmask, parameters: parameters)
    }

    public func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: MLXContext<Profile>
    ) async -> Result<EdgeToolsEngineGeneration, any Error> {
      var state = state
      let metadata = state.model.finish()

      let generation: EdgeToolsEngineGeneration?
      let finalResult: Result<EdgeToolsEngineGeneration, any Error>
      switch result {
      case .success(var value):
        state.model.commitGeneration(stopTokenIds: self.generationLoop.stopTokenIds)
        value.metadata.merge(metadata) { _, finalValue in finalValue }
        generation = value
        finalResult = .success(value)
      case .failure(let error):
        await state.model.resetGeneration()
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
      try self.validate(context)
      return try await self.prefill(
        snapshot: context.begin(appending: promptPrefix),
        tools: tools,
        context: context
      )
    }

    public func prefill(
      tools: [EdgeToolDefinition] = [],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try await self.prefill(
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
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
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

    private func generationTask(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel,
      makeState: @escaping @Sendable () async throws -> ModelGenerationState
    ) -> AnyGenerationTask {
      AnyGenerationTask { stopper in
        var state = try await makeState()
        var preparedPrompt = prompt
        let result: Result<EdgeToolsEngineGeneration, any Error>
        do {
          result = .success(
            try await self.generationLoop.run(
              state: &state,
              stopper: stopper,
              channel: channel,
              grammarEngine: self.grammarEngine,
              maximumTokenCount: parameters.maxTokens,
              grammar: { state in
                try self.grammar(
                  prompt: preparedPrompt,
                  tools: tools,
                  parameters: parameters,
                  state: state
                )
              },
              prepare: { parser, state in
                return try await self.prepare(
                  prompt: &preparedPrompt,
                  tools: tools,
                  parameters: parameters,
                  parser: &parser,
                  state: &state
                )
              },
              decode: { bitmask, state in
                try await self.decode(
                  bitmask: bitmask,
                  parameters: parameters,
                  state: &state
                )
              }
            )
          )
        } catch {
          result = .failure(error)
        }
        return try await self.finalize(state: state, result: result, context: context).get()
      }
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

    private func validate(_ context: MLXContext<Profile>) throws {
      guard context.engineIdentity === self.identity else {
        throw EdgeToolsError.incompatibleContext
      }
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
