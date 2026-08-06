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

  // MARK: - EdgeTools MLX Model

  public protocol EdgeToolsMLXModel: LanguageModel, SendableMetatype {
    associatedtype ModelConfiguration: Decodable
    associatedtype Prompt: Sendable
    associatedtype GenerateParameters: EdgeToolsMLXGenerateParameters =
      DefaultEdgeToolsMLXGenerateParameters
    associatedtype ToolCallParser: EdgeToolCallParser

    var vocabularySize: Int { get }

    func grammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar

    func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar

    func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> LMInput
  }

  extension EdgeToolsMLXModel
  where
    GenerateParameters: EdgeToolsConstrainedGenerateParameters,
    GenerateParameters.Constraint.Grammar == XGRGrammar,
    GenerateParameters.Constraint.Context == XGRGrammarContext
  {
    public func grammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      let constraint = parameters.constraint
      let toolCallGrammar = try constraint.toolCallRange.map {
        try self.toolCallGrammar(tools: tools, range: $0)
      }
      return try constraint.grammar(
        toolCallGrammar: toolCallGrammar,
        context: context
      )
    }
  }

  extension EdgeToolsMLXModel {
    package static func loadEdgeToolsLanguageModel(
      from directoryURL: URL,
      model makeModel: @Sendable (ModelConfiguration) throws -> Self
    ) throws -> Self {
      let configuration = try decodeModelConfiguration(
        ModelConfiguration.self,
        in: directoryURL
      )
      let baseConfiguration = try decodeModelConfiguration(
        BaseConfiguration.self,
        in: directoryURL
      )
      guard let configuration, let baseConfiguration else {
        throw EdgeToolsError.failedToLoadConfiguration
      }

      let model = try makeModel(configuration)
      try loadWeights(
        modelDirectory: directoryURL,
        model: model,
        perLayerQuantization: baseConfiguration.perLayerQuantization
      )
      return model
    }
  }

  public struct EdgeToolsMLXError: Hashable, Sendable, Error {
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

  // MARK: - EdgeToolsMLXGenerateParameters

  public protocol EdgeToolsMLXGenerateParameters: EdgeToolsEngineGenerateParameters {
    var sampler: any LogitSampler { get }
    var processor: (any LogitProcessor)? { get }
    var kvCacheQuantizationBits: Int? { get }
    var kvCacheQuantizationGroupSize: Int { get }
    var quantizedKVStart: Int { get }
    var synchronizeStreamForMemorySnapshots: Bool { get }
  }

  public struct DefaultEdgeToolsMLXGenerateParameters:
    EdgeToolsMLXGenerateParameters,
    EdgeToolsConstrainedGenerateParameters
  {
    public static var `default`: Self { Self() }

    public var sampler: any LogitSampler
    public var processor: (any LogitProcessor)?
    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?
    public var kvCacheQuantizationBits: Int?
    public var kvCacheQuantizationGroupSize: Int
    public var quantizedKVStart: Int
    public var synchronizeStreamForMemorySnapshots: Bool

    public init(
      sampler: any LogitSampler = CategoricalSampler(temperature: 0.6),
      processor: (any LogitProcessor)? = nil,
      constraint: EdgeToolsXGRGenerationConstraint = .unconstrained,
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

  // MARK: - EdgeToolsMLXEngine

  public typealias EdgeToolsMLXEngine<Model: EdgeToolsMLXModel> =
    EdgeToolsModelEngine<_EdgeToolsMLXModel<Model>>

  // MARK: - Prompt Conversion

  #if Transformers && canImport(Tokenizers)
    extension EdgeToolsMLXModel
    where Self: LLMModel, Prompt == EdgeToolsLLMPrompt {
      public func input(
        prompt: EdgeToolsLLMPrompt,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsXGRTokenizer
      ) throws -> LMInput {
        guard let tokenizer = tokenizer as? EdgeToolsPreTrainedTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let tokenIds = try tokenizer.tokenizer.applyChatTemplate(
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

  // MARK: - EdgeToolsMLXModel Adapter

  // swiftlint:disable:next type_name
  public struct _EdgeToolsMLXModel<Model: EdgeToolsMLXModel>: EdgeToolsModel {
    public typealias Prompt = Model.Prompt
    public typealias Input = LMInput
    public typealias Tokenizer = AnyEdgeToolsXGRTokenizer
    public typealias GenerateParameters = Model.GenerateParameters
    public typealias ToolCallParser = Model.ToolCallParser
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

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
    private var cachedPrefill: CachedPrefill?
    private var generation: Generation?

    public init(model: Model) {
      self.model = model
    }

    public var vocabularySize: Int { self.model.vocabularySize }

    public func makeGrammarContext(
      tokenizer: AnyEdgeToolsXGRTokenizer
    ) throws -> XGRGrammarContext {
      try XGRGrammarContext(
        tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: self.vocabularySize)
      )
    }

    public func makeGrammarCompiler(
      context: borrowing XGRGrammarContext
    ) throws -> XGRCompiler {
      try XGRCompiler(tokenizerInfo: context.tokenizerInfo)
    }

    public func grammar(
      tools: [EdgeToolDefinition],
      parameters: Model.GenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      try self.model.grammar(tools: tools, parameters: parameters, context: context)
    }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try self.model.toolCallGrammar(tools: tools, range: range)
    }

    public func input(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: AnyEdgeToolsXGRTokenizer
    ) throws -> EdgeToolsModelInput<LMInput> {
      let input = try self.model.input(prompt: prompt, tools: tools, tokenizer: tokenizer)
      return EdgeToolsModelInput(
        value: input,
        tokenIds: input.text.tokens.asArray(EdgeToolsToken.ID.self)
      )
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
        let output = self.model(
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
      let cache = self.model.newCache(parameters: nil)
      let start = clock.now
      let output = try self.edgeToolsPrepare(input: input, cache: cache)
      eval(output.logits)
      eval(cache)
      self.cachedPrefill = CachedPrefill(
        tokenIds: tokenIds,
        cache: cache.map { $0.copy() },
        output: output
      )
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
      guard let cachedPrefill = self.cachedPrefill,
        tokenIds.starts(with: cachedPrefill.tokenIds)
      else {
        let cache = self.model.newCache(parameters: nil)
        return (try self.edgeToolsPrepare(input: input, cache: cache), cache, tokenIds.count)
      }
      let suffixCount = tokenIds.count - cachedPrefill.tokenIds.count
      let cache = cachedPrefill.cache.map { $0.copy() }
      let output =
        if suffixCount == 0 {
          cachedPrefill.output
        } else {
          self.model(
            input.text[cachedPrefill.tokenIds.count...][text: .newAxis],
            cache: cache,
            state: cachedPrefill.output.state
          )
        }
      return (output, cache, suffixCount)
    }

    private func edgeToolsPrepare(input: LMInput, cache: [any KVCache]) throws -> LMOutput {
      switch try self.model.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw EdgeToolsMLXError(code: .emptyInput, message: "Model received empty input.")
        }
        return self.model(tokens[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil)
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
    public init<MLXModel: EdgeToolsMLXModel>(
      model: sending MLXModel,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws where Model == _EdgeToolsMLXModel<MLXModel> {
      try self.init(
        model: _EdgeToolsMLXModel(model: model),
        tokenizer: AnyEdgeToolsXGRTokenizer(tokenizer)
      )
    }

    public init<MLXModel: EdgeToolsMLXModel>(
      from directoryURL: URL,
      model makeModel: @Sendable (MLXModel.ModelConfiguration) throws -> MLXModel
    ) async throws where Model == _EdgeToolsMLXModel<MLXModel> {
      try await self.init(
        loading: MLXModel.self,
        from: directoryURL,
        model: makeModel
      )
    }

    private init<MLXModel: EdgeToolsMLXModel>(
      loading modelType: MLXModel.Type,
      from directoryURL: URL,
      model makeModel: @Sendable (MLXModel.ModelConfiguration) throws -> MLXModel
    ) async throws where Model == _EdgeToolsMLXModel<MLXModel> {
      let tokenizer = try await loadEdgeToolsTokenizer(from: directoryURL)
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let model = try MLXModel.loadEdgeToolsLanguageModel(from: directoryURL, model: makeModel)
      try self.init(
        model: _EdgeToolsMLXModel(model: model),
        tokenizer: AnyEdgeToolsXGRTokenizer(tokenizer)
      )
    }
  }
#endif
