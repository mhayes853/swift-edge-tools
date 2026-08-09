#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN

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
    associatedtype ToolCallParser: EdgeToolCallParser
    associatedtype GrammarContext = Void
    associatedtype GrammarCompiler: EdgeToolsGrammarCompiler, ~Copyable
    where GrammarCompiler.Context == GrammarContext

    static var extraStopTokens: Set<String> { get }

    static func grammarContext(
      tokenizer: any EdgeToolsTokenizer,
      vocabularySize: Int,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> GrammarContext
    static func grammarCompiler(context: borrowing GrammarContext) throws -> GrammarCompiler

    static func grammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      context: GrammarContext
    ) throws -> GrammarCompiler.Grammar

    static func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> GrammarCompiler.Grammar

    static nonisolated(nonsending) func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput
  }

  public protocol MLXLLMModelProfile: MLXModelProfile {}

  #if canImport(CoreImage) && canImport(MLXVLM)
    public protocol MLXVLMModelProfile: MLXModelProfile {}
  #endif

  extension MLXModelProfile
  where
    GenerateParameters: EdgeToolsConstrainedGenerateParameters,
    GenerateParameters.Constraint.Grammar == GrammarCompiler.Grammar,
    GenerateParameters.Constraint.Context == GrammarContext
  {
    public static func grammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      context: GrammarContext
    ) throws -> GrammarCompiler.Grammar {
      let constraint = parameters.constraint
      let toolCallGrammar = try constraint.toolCallRange.map {
        try Self.toolCallGrammar(tools: tools, range: $0)
      }
      return try constraint.grammar(toolCallGrammar: toolCallGrammar, context: context)
    }
  }

  extension MLXModelProfile {
    public static var extraStopTokens: Set<String> { [] }
  }

  extension LMInput: EdgeToolsModelInput {
    public var tokenIds: [EdgeToolsToken.ID] {
      self.text.tokens.asArray(EdgeToolsToken.ID.self)
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
    var sampler: any LogitSampler { get }
    var processor: (any LogitProcessor)? { get }
    var kvCacheQuantizationBits: Int? { get }
    var kvCacheQuantizationGroupSize: Int { get }
    var quantizedKVStart: Int { get }
    var synchronizeStreamForMemorySnapshots: Bool { get }
  }

  // MARK: - DefaultMLXGenerateParameters

  #if XGrammar
    public struct DefaultMLXGenerateParameters:
      MLXGenerateParameters,
      EdgeToolsConstrainedGenerateParameters
    {
      public static var `default`: Self { Self() }

      public var sampler: any LogitSampler
      public var processor: (any LogitProcessor)?
      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int
      public var quantizedKVStart: Int
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: any LogitSampler = CategoricalSampler(temperature: 0.6),
        processor: (any LogitProcessor)? = nil,
        constraint: XGRGenerationConstraint = .unconstrained,
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self.sampler = sampler
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

    extension MLXModelProfile
    where GrammarCompiler == XGRCompiler, GrammarContext == XGRGrammarContext {
      public static func grammarContext(
        tokenizer: any EdgeToolsTokenizer,
        vocabularySize: Int,
        stopTokenIds: Set<EdgeToolsToken.ID>
      ) throws -> XGRGrammarContext {
        guard let tokenizer = tokenizer as? any XGRTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        return try XGRGrammarContext(
          tokenizerInfo: tokenizer.tokenizerInfo(
            modelVocabularySize: vocabularySize,
            extraStopTokenIds: stopTokenIds
          )
        )
      }

      public static func grammarCompiler(
        context: borrowing XGRGrammarContext
      ) throws -> XGRCompiler {
        try XGRCompiler(tokenizerInfo: context.tokenizerInfo)
      }
    }
  #endif

  // MARK: - MLXEngine

  public typealias MLXEngine<Profile: MLXModelProfile> =
    EdgeToolsModelEngine<EdgeToolsMLXModel<Profile>>

  // MARK: - Prompt Conversion

  #if canImport(Tokenizers)
    extension MLXLLMModelProfile where Prompt == EdgeToolsLLMPrompt {
      public static nonisolated(nonsending) func input(
        prompt: EdgeToolsLLMPrompt,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        processor _: (any UserInputProcessor)?
      ) async throws -> LMInput {
        guard let tokenizer = tokenizer as? TransformersTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let tokenIds = try tokenizer.base.applyChatTemplate(
          messages: try prompt.mlxMessages(),
          tools: tools.mlxToolSpecs,
          additionalContext: nil
        )
        return LMInput(tokens: MLXArray(tokenIds))
      }
    }

    extension EdgeToolsLLMPrompt {
      fileprivate func mlxMessages() throws -> [MLXLMCommon.Message] {
        try self.messages.map { try $0.mlxMessage() }
      }
    }

    extension EdgeToolsLLMPrompt.Message {
      public func mlxMessage() throws -> MLXLMCommon.Message {
        switch self {
        case .system(let content):
          ["role": "system", "content": content]
        case .user(let content, images: _, videos: _, audio: _):
          ["role": "user", "content": content]
        case .assistant(let content, let toolCalls):
          self.mlxAssistantMessage(content: content, toolCalls: toolCalls)
        case .tool(let name, let response):
          [
            "role": "tool",
            "content": String(decoding: try Self.encode(response), as: UTF8.self),
            "name": name
          ]
        }
      }

      private func mlxAssistantMessage(
        content: String?,
        toolCalls: [EdgeRawToolCall]
      ) -> MLXLMCommon.Message {
        var message: MLXLMCommon.Message = ["role": "assistant"]
        if let content {
          message["content"] = content
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
      package var mlxToolSpecs: [ToolSpec]? {
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

      init<Assets: Sequence>(assets: Assets) throws where Assets.Element == EdgeToolsLLMPrompt.Asset {
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
          temporaryURLs.forEach { try? FileManager.default.removeItem(at: $0) }
          throw error
        }

        self.videos = videos
        self.temporaryURLs = temporaryURLs
      }

      deinit {
        self.temporaryURLs.forEach { try? FileManager.default.removeItem(at: $0) }
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

    extension EdgeToolsLLMPrompt.Asset {
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

    extension Sequence where Element == EdgeToolsLLMPrompt.Asset {
      package func mlxImages() throws -> [UserInput.Image] {
        try self.map { try $0.mlxImage() }
      }
    }

    extension EdgeToolsLLMPrompt {
      package func rejectAudio() throws {
        guard !self.audio.isEmpty else { return }
        throw EdgeToolsError.unsupportedMedia(
          "This MLX model integration does not support audio input."
        )
      }

      package func rejectVideos() throws {
        guard !self.videos.isEmpty else { return }
        throw EdgeToolsError.unsupportedMedia(
          "This MLX model integration does not support video input."
        )
      }

      public func mlxUserInput(
        tools: [EdgeToolDefinition],
        videos: [UserInput.Video] = [],
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) throws -> UserInput {
        try self.rejectAudio()
        if videos.isEmpty { try self.rejectVideos() }
        return UserInput(
          messages: try self.messages.map(transformMessage),
          images: try self.images.mlxImages(),
          videos: videos,
          tools: tools.mlxToolSpecs
        )
      }

      public func mlxVLMInput(
        tools: [EdgeToolDefinition],
        processor: any UserInputProcessor,
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) async throws -> LMInput {
        let videoInputs = try MLXTemporaryVideoInputs(assets: self.videos)
        return try await processor.prepare(
          input: try self.mlxUserInput(
            tools: tools,
            videos: videoInputs.videos
          ) { try transformMessage($0) }
        )
      }
    }
  #endif

  // MARK: - MLXModel Adapter

  public struct EdgeToolsMLXModel<Profile: MLXModelProfile>: EdgeToolsModel {
    public typealias Prompt = Profile.Prompt
    public typealias Input = LMInput
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias ToolCallParser = Profile.ToolCallParser
    public typealias GrammarCompiler = Profile.GrammarCompiler
    public typealias GrammarContext = Profile.GrammarContext

    private struct CachedPrefill {
      let input: LMInput
      let tokenIds: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
      let context: EdgeToolsLLMPrefillContext?
    }

    private final class PrefillCacheState {
      var cachedPrefill: CachedPrefill?
      var inputContext: EdgeToolsLLMPrefillContext?

      func input(for context: EdgeToolsLLMPrefillContext) -> LMInput? {
        self.inputContext = context
        guard let cachedPrefill, cachedPrefill.context == context else { return nil }
        return cachedPrefill.input
      }

      func clearInputContext() {
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
      let synchronizeStreamForMemorySnapshots: Bool
      let generationStartSnapshot: Memory.Snapshot
      let postPrefillSnapshot: Memory.Snapshot
    }

    private let vocabularySizeValue: Int
    private var languageModel: any LanguageModel
    private let processor: (any UserInputProcessor)?
    private let configuredExtraStopTokenIds: Set<EdgeToolsToken.ID>
    private let prefillCacheState = PrefillCacheState()
    private var generation: Generation?

    package init(
      languageModel: any LanguageModel,
      processor: (any UserInputProcessor)? = nil,
      vocabularySize: Int,
      extraStopTokenIds: Set<EdgeToolsToken.ID>
    ) {
      self.languageModel = languageModel
      self.processor = processor
      self.vocabularySizeValue = vocabularySize
      self.configuredExtraStopTokenIds = extraStopTokenIds
    }

    public var vocabularySize: Int { self.vocabularySizeValue }
    public var extraStopTokenIds: Set<EdgeToolsToken.ID> { self.configuredExtraStopTokenIds }

    public func grammarContext(tokenizer: any EdgeToolsTokenizer) throws -> Profile.GrammarContext {
      var stopTokenIds = self.extraStopTokenIds
      if let eosTokenId = tokenizer.eosTokenId { stopTokenIds.insert(eosTokenId) }
      return try Profile.grammarContext(
        tokenizer: tokenizer,
        vocabularySize: self.vocabularySize,
        stopTokenIds: stopTokenIds
      )
    }

    public func grammarCompiler(
      context: borrowing Profile.GrammarContext
    ) throws -> Profile.GrammarCompiler {
      try Profile.grammarCompiler(context: context)
    }

    public func grammar(
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      context: Profile.GrammarContext
    ) throws -> Profile.GrammarCompiler.Grammar {
      try Profile.grammar(tools: tools, parameters: parameters, context: context)
    }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> Profile.GrammarCompiler.Grammar {
      try Profile.toolCallGrammar(tools: tools, range: range)
    }

    public nonisolated(nonsending) func input(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> LMInput {
      if let prompt = prompt as? EdgeToolsLLMPrompt {
        let context = EdgeToolsLLMPrefillContext(prompt: prompt, tools: tools)
        if let input = self.prefillCacheState.input(for: context) { return input }
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

    public nonisolated(nonsending) mutating func prepare(
      input: LMInput,
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
      let token = parameters.sampler.sample(logits: maskedLogits)
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

    public mutating func resetGeneration() {
      self.generation = nil
    }

    public nonisolated(nonsending) mutating func prefill(
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

  extension EdgeToolsMLXModel: EdgeToolsPrefillableModel {}

  extension EdgeToolsModelEngine {
    public init<Profile: MLXModelProfile>(
      languageModel: sending any LanguageModel,
      tokenizer: sending any EdgeToolsTokenizer,
      processor: sending (any UserInputProcessor)? = nil,
      vocabularySize: Int,
      extraStopTokenIds: Set<EdgeToolsToken.ID> = []
    ) throws where Model == EdgeToolsMLXModel<Profile> {
      try self.init(
        model: EdgeToolsMLXModel(
          languageModel: languageModel,
          processor: processor,
          vocabularySize: vocabularySize,
          extraStopTokenIds: extraStopTokenIds
        ),
        tokenizer: tokenizer
      )
    }

    public init<Profile: MLXLLMModelProfile>(
      from directoryURL: URL,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void = { _, _ in }
    ) async throws where Model == EdgeToolsMLXModel<Profile> {
      try await self.init(
        from: MLXModelDirectory(url: directoryURL),
        patchWeights: patchWeights
      )
    }

    public init<Profile: MLXLLMModelProfile>(
      from directory: MLXModelDirectory,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void = { _, _ in }
    ) async throws where Model == EdgeToolsMLXModel<Profile> {
      let configurationData = try directory.loadConfigurationData()
      let baseConfiguration = try JSONDecoder.json5()
        .decode(
          BaseConfiguration.self,
          from: configurationData
        )
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
      public init<Profile: MLXVLMModelProfile>(
        from directoryURL: URL,
        patchWeights: (
          _ weights: inout [String: MLXArray],
          _ model: any LanguageModel
        ) throws -> Void = { _, _ in }
      ) async throws where Model == EdgeToolsMLXModel<Profile> {
        try await self.init(
          from: MLXModelDirectory(url: directoryURL),
          patchWeights: patchWeights
        )
      }

      public init<Profile: MLXVLMModelProfile>(
        from directory: MLXModelDirectory,
        patchWeights: (
          _ weights: inout [String: MLXArray],
          _ model: any LanguageModel
        ) throws -> Void = { _, _ in }
      ) async throws where Model == EdgeToolsMLXModel<Profile> {
        let configurationData = try directory.loadConfigurationData()
        let baseConfiguration = try JSONDecoder.json5()
          .decode(
            BaseConfiguration.self,
            from: configurationData
          )
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

    private init<Profile: MLXModelProfile>(
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
    ) throws where Model == EdgeToolsMLXModel<Profile> {
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
        extraStopTokenIds: extraStopTokenIds
      )
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
    profile _: Profile.Type,
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
      mlxArraysEqual(lhs.features, rhs.features)
        && mlxArraysEqual(lhs.mask, rhs.mask)
    default:
      false
    }
  }

  private func mlxArraysEqual(_ lhs: MLXArray?, _ rhs: MLXArray?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      lhs.dtype == rhs.dtype
        && lhs.shape == rhs.shape
        && lhs.arrayEqual(rhs).item(Bool.self)
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
          .allSatisfy { lhs, rhs in
            lhs.t == rhs.t && lhs.h == rhs.h && lhs.w == rhs.w
          }
    default:
      false
    }
  }
#endif
