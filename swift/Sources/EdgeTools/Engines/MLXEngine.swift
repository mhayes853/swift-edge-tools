#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  #if Transformers && canImport(Tokenizers)
    import Tokenizers
  #endif

  #if Transformers && canImport(CoreImage) && canImport(MLXVLM)
    import CoreImage
    import Foundation
    import MLXVLM
  #endif

  // MARK: - MLXModel

  public protocol MLXModel: SendableMetatype {
    associatedtype UnderlyingModel: LanguageModel
    associatedtype ModelConfiguration
    associatedtype Prompt: Sendable
    associatedtype GenerateParameters: MLXGenerateParameters
    associatedtype ToolCallParser: EdgeToolCallParser
    associatedtype GrammarContext = Void
    associatedtype GrammarCompiler: EdgeToolsGrammarCompiler, ~Copyable
    where GrammarCompiler.Context == GrammarContext

    init(configuration: ModelConfiguration) throws

    var languageModel: UnderlyingModel { get }
    var vocabularySize: Int { get }
    var extraStopTokens: Set<String> { get }

    func loadWeights(from directoryURL: URL) throws

    func grammarContext(tokenizer: any EdgeToolsTokenizer) throws -> GrammarContext
    func grammarCompiler(context: borrowing GrammarContext) throws -> GrammarCompiler

    func grammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      context: GrammarContext
    ) throws -> GrammarCompiler.Grammar

    func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> GrammarCompiler.Grammar

    nonisolated(nonsending) func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsModelInput<LMInput>
  }

  extension MLXModel
  where
    GenerateParameters: EdgeToolsConstrainedGenerateParameters,
    GenerateParameters.Constraint.Grammar == GrammarCompiler.Grammar,
    GenerateParameters.Constraint.Context == GrammarContext
  {
    public func grammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      context: GrammarContext
    ) throws -> GrammarCompiler.Grammar {
      let constraint = parameters.constraint
      let toolCallGrammar = try constraint.toolCallRange.map {
        try self.toolCallGrammar(tools: tools, range: $0)
      }
      return try constraint.grammar(toolCallGrammar: toolCallGrammar, context: context)
    }
  }

  extension MLXModel {
    public var extraStopTokens: Set<String> { [] }

    public func loadWeights(from directoryURL: URL) throws {
      let baseConfiguration = try decodeModelConfiguration(
        BaseConfiguration.self,
        in: directoryURL,
        decoder: JSONDecoder.json5()
      )
      guard let baseConfiguration else {
        throw EdgeToolsError.failedToLoadConfiguration
      }

      try MLXLMCommon.loadWeights(
        modelDirectory: directoryURL,
        model: self.languageModel,
        perLayerQuantization: baseConfiguration.perLayerQuantization
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

    // MARK: - MLXModel + XGrammar

    extension MLXModel
    where GrammarCompiler == XGRCompiler, GrammarContext == XGRGrammarContext {
      public func grammarContext(tokenizer: any EdgeToolsTokenizer) throws -> XGRGrammarContext {
        guard let tokenizer = tokenizer as? any XGRTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        return try XGRGrammarContext(
          tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: self.vocabularySize)
        )
      }

      public func grammarCompiler(context: borrowing XGRGrammarContext) throws -> XGRCompiler {
        try XGRCompiler(tokenizerInfo: context.tokenizerInfo)
      }
    }
  #endif

  // MARK: - MLXEngine

  public typealias MLXEngine<Model: MLXModel> =
    EdgeToolsModelEngine<_EdgeToolsMLXModel<Model>>

  // MARK: - Prompt Conversion

  #if Transformers && canImport(Tokenizers)
    extension MLXModel
    where UnderlyingModel: LLMModel, Prompt == EdgeToolsLLMPrompt {
      public nonisolated(nonsending) func input(
        prompt: EdgeToolsLLMPrompt,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer
      ) async throws -> EdgeToolsModelInput<LMInput> {
        guard let tokenizer = tokenizer as? TransformersTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let tokenIds = try tokenizer.base.applyChatTemplate(
          messages: try prompt.mlxMessages(),
          tools: tools.mlxToolSpecs,
          additionalContext: nil
        )
        return EdgeToolsModelInput(
          value: LMInput(tokens: MLXArray(tokenIds)),
          tokenIds: tokenIds
        )
      }
    }

    extension EdgeToolsLLMPrompt {
      fileprivate func mlxMessages() throws -> [MLXLMCommon.Message] {
        try self.messages.map { try $0.mlxMessage() }
      }
    }

    extension EdgeToolsLLMPrompt.Message {
      package func mlxMessage() throws -> MLXLMCommon.Message {
        switch self {
        case .system(let content):
          ["role": "system", "content": content]
        case .user(let content, images: _, audio: _):
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

    extension Sequence where Element == EdgeToolDefinition {
      package var mlxToolSpecs: [ToolSpec]? {
        let specifications = self.compactMap { definition -> ToolSpec? in
          guard definition.includesSchemaInInstructions else { return nil }
          return [
            "type": "function",
            "function": [
              "name": definition.name,
              "description": definition.description,
              "parameters": definition.arguments.edgeToolsValue.mlxValue
            ] as MLXLMCommon.Message
          ]
        }
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

  #if Transformers && canImport(CoreImage) && canImport(MLXVLM)
    package func decodeMLXVLMProcessorConfiguration<Configuration: Decodable>(
      _ type: Configuration.Type,
      in directoryURL: URL
    ) throws -> Configuration? {
      let preprocessorURL = directoryURL.appending(path: "preprocessor_config.json")
      let filename =
        FileManager.default.fileExists(atPath: preprocessorURL.path)
        ? "preprocessor_config.json"
        : "processor_config.json"
      return try decodeModelConfiguration(type, named: filename, in: directoryURL)
    }

    extension EdgeToolsLLMPrompt.Asset {
      package func mlxImage() throws -> UserInput.Image {
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

      package func mlxVLMUserInput(
        tools: [EdgeToolDefinition],
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) throws -> UserInput {
        try self.rejectAudio()
        return UserInput(
          messages: try self.messages.map(transformMessage),
          images: try self.images.mlxImages(),
          tools: tools.mlxToolSpecs
        )
      }
    }
  #endif

  // MARK: - MLXModel Adapter

  public struct _EdgeToolsMLXModel<Model: MLXModel>: EdgeToolsModel {
    public typealias Prompt = Model.Prompt
    public typealias Input = LMInput
    public typealias GenerateParameters = Model.GenerateParameters
    public typealias ToolCallParser = Model.ToolCallParser
    public typealias GrammarCompiler = Model.GrammarCompiler
    public typealias GrammarContext = Model.GrammarContext

    private struct CachedPrefill {
      let tokenIds: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
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

    private var model: Model
    private let configuredExtraStopTokenIds: Set<EdgeToolsToken.ID>
    private var cachedPrefill: CachedPrefill?
    private var generation: Generation?

    public init(model: Model) {
      self.model = model
      self.configuredExtraStopTokenIds = []
    }

    package init(
      model: Model,
      extraStopTokenIds: Set<EdgeToolsToken.ID>
    ) {
      self.model = model
      self.configuredExtraStopTokenIds = extraStopTokenIds
    }

    public var vocabularySize: Int { self.model.vocabularySize }
    public var extraStopTokenIds: Set<EdgeToolsToken.ID> { self.configuredExtraStopTokenIds }

    public func grammarContext(tokenizer: any EdgeToolsTokenizer) throws -> Model.GrammarContext {
      try self.model.grammarContext(tokenizer: tokenizer)
    }

    public func grammarCompiler(
      context: borrowing Model.GrammarContext
    ) throws -> Model.GrammarCompiler {
      try self.model.grammarCompiler(context: context)
    }

    public func grammar(
      tools: [EdgeToolDefinition],
      parameters: Model.GenerateParameters,
      context: Model.GrammarContext
    ) throws -> Model.GrammarCompiler.Grammar {
      try self.model.grammar(tools: tools, parameters: parameters, context: context)
    }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> Model.GrammarCompiler.Grammar {
      try self.model.toolCallGrammar(tools: tools, range: range)
    }

    public nonisolated(nonsending) func input(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsModelInput<LMInput> {
      try await self.model.input(prompt: prompt, tools: tools, tokenizer: tokenizer)
    }

    public nonisolated(nonsending) mutating func prepare(
      input: LMInput,
      parameters: Model.GenerateParameters
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
      parameters: Model.GenerateParameters
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
        let output = self.model.languageModel(
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
      let cache = self.model.languageModel.newCache(parameters: nil)
      let start = clock.now
      let output = try self.prepareModelOutput(input: input, cache: cache)
      eval(output.logits)
      eval(cache)
      self.cachedPrefill =
        if input.image == nil, input.video == nil {
          CachedPrefill(
            tokenIds: tokenIds,
            cache: cache.map { $0.copy() },
            output: output
          )
        } else {
          nil
        }
      let snapshot = Self.memorySnapshot(synchronize: true)
      var metadata = EdgeToolsMetadata()
      metadata.mlxEnginePostPrefillMemorySnapshot = snapshot
      return EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(
          tokens: tokenIds.count,
          duration: start.duration(to: clock.now)
        ),
        metadata: metadata
      )
    }

    private func preparedOutput(
      input: LMInput,
      tokenIds: [EdgeToolsToken.ID]
    ) throws -> (output: LMOutput, cache: [any KVCache], tokenCount: Int) {
      guard input.image == nil, input.video == nil,
        let cachedPrefill = self.cachedPrefill,
        tokenIds.starts(with: cachedPrefill.tokenIds)
      else {
        let cache = self.model.languageModel.newCache(parameters: nil)
        return (try self.prepareModelOutput(input: input, cache: cache), cache, tokenIds.count)
      }
      let suffixCount = tokenIds.count - cachedPrefill.tokenIds.count
      let cache = cachedPrefill.cache.map { $0.copy() }
      let output =
        if suffixCount == 0 {
          cachedPrefill.output
        } else {
          self.model.languageModel(
            input.text[cachedPrefill.tokenIds.count...][text: .newAxis],
            cache: cache,
            state: cachedPrefill.output.state
          )
        }
      return (output, cache, suffixCount)
    }

    private func prepareModelOutput(input: LMInput, cache: [any KVCache]) throws -> LMOutput {
      switch try self.model.languageModel.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw MLXEngineError(code: .emptyInput, message: "Model received empty input.")
        }
        return self.model.languageModel(
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

  extension _EdgeToolsMLXModel: EdgeToolsPrefillableModel {}

  extension EdgeToolsModelEngine {
    public init<Base: MLXModel>(
      model: sending Base,
      tokenizer: sending any EdgeToolsTokenizer
    ) throws where Model == _EdgeToolsMLXModel<Base> {
      try self.init(model: _EdgeToolsMLXModel(model: model), tokenizer: tokenizer)
    }

    public init<Base: MLXModel>(
      from directoryURL: URL
    ) async throws
    where
      Model == _EdgeToolsMLXModel<Base>,
      Base.ModelConfiguration: Decodable
    {
      guard
        let configuration = try decodeModelConfiguration(
          Base.ModelConfiguration.self,
          in: directoryURL,
          decoder: JSONDecoder.json5()
        )
      else {
        throw EdgeToolsError.failedToLoadConfiguration
      }
      try await self.init(
        from: directoryURL,
        configuration: configuration
      )
    }

    public init<Base: MLXModel>(
      from directoryURL: URL,
      configuration: Base.ModelConfiguration
    ) async throws where Model == _EdgeToolsMLXModel<Base> {
      let tokenizer = try await EdgeToolsAutoTokenizer.from(modelDirectory: directoryURL)
      let model = try Base(configuration: configuration)
      try model.loadWeights(from: directoryURL)
      var extraStopTokenIds = try loadMLXExtraStopTokenIds(from: directoryURL)
      extraStopTokenIds.formUnion(
        model.extraStopTokens.compactMap { tokenizer.convertTokenToId($0) }
      )
      if let eosTokenId = tokenizer.eosTokenId { extraStopTokenIds.remove(eosTokenId) }
      try self.init(
        model: _EdgeToolsMLXModel(
          model: model,
          extraStopTokenIds: extraStopTokenIds
        ),
        tokenizer: tokenizer
      )
    }
  }

  private func loadMLXExtraStopTokenIds(from directoryURL: URL) throws -> Set<EdgeToolsToken.ID> {
    let baseConfiguration = try decodeModelConfiguration(
      BaseConfiguration.self,
      in: directoryURL,
      decoder: JSONDecoder.json5()
    )
    var extraStopTokenIds = Set(baseConfiguration?.eosTokenIds?.values ?? [])
    let generationConfigurationURL = directoryURL.appending(path: "generation_config.json")
    if let data = try? Data(contentsOf: generationConfigurationURL),
      let configuration = try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: data),
      let values = configuration.eosTokenIds?.values
    {
      extraStopTokenIds = Set(values)
    }
    return extraStopTokenIds
  }
#endif
