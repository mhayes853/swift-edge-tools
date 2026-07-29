#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  public import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  #if Transformers
    import Tokenizers
  #endif

  // MARK: - EdgeTools MLX Model

  public protocol EdgeToolsMLXModel: EdgeToolsModel, LanguageModel
  where
    Input == LMInput,
    Logits == MLXArray,
    Assets == EdgeToolsMLXModelAssets,
    GenerateParameters == EdgeToolsMLXGenerateParameters,
    GenerationState == EdgeToolsMLXGenerationState
  {
    associatedtype ModelConfiguration: Decodable
  }

  extension EdgeToolsMLXModel {
    package static func loadEdgeToolsLanguageModel(
      from directoryURL: URL,
      model makeModel: (ModelConfiguration) throws -> Self
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

  public struct EdgeToolsMLXGenerateParameters: EdgeToolsModelEngineGenerateParameters {
    public static var `default`: Self { Self() }

    private var makeSampler: @Sendable () -> any LogitSampler
    private var makeProcessor: @Sendable () -> (any LogitProcessor)?

    public var sampler: any LogitSampler { self.makeSampler() }
    public var processor: (any LogitProcessor)? { self.makeProcessor() }

    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?
    public var kvCacheQuantizationBits: Int?
    public var kvCacheQuantizationGroupSize: Int
    public var quantizedKVStart: Int
    public var synchronizeStreamForMemorySnapshots: Bool

    public init(
      sampler: @autoclosure @escaping @Sendable () -> any LogitSampler = ArgMaxSampler(),
      processor: @autoclosure @escaping @Sendable () -> (any LogitProcessor)? = nil,
      constraint: EdgeToolsXGRGenerationConstraint = .tools,
      maxTokens: Int? = 1024,
      kvCacheQuantizationBits: Int? = nil,
      kvCacheQuantizationGroupSize: Int = 64,
      quantizedKVStart: Int = 0,
      synchronizeStreamForMemorySnapshots: Bool = true
    ) {
      self.makeSampler = sampler
      self.makeProcessor = processor
      self.constraint = constraint
      self.maxTokens = maxTokens
      self.kvCacheQuantizationBits = kvCacheQuantizationBits
      self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
      self.quantizedKVStart = quantizedKVStart
      self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
    }
  }

  // MARK: - EdgeToolsMLXModelAssets

  public final class EdgeToolsMLXModelAssets {
    struct CachedPrefill {
      let tokenIds: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
    }

    var cachedPrefill: CachedPrefill?

    public init() {}
  }

  // MARK: - EdgeToolsMLXGenerationState

  public struct EdgeToolsMLXGenerationState {
    var cache: [any KVCache]
    var outputState: LMOutput.State?
    let sampler: any LogitSampler
    var processor: (any LogitProcessor)?
    let kvCacheQuantizationBits: Int?
    let kvCacheQuantizationGroupSize: Int
    let quantizedKVStart: Int
    let synchronizeStreamForMemorySnapshots: Bool
    let generationStartSnapshot: Memory.Snapshot
    let postPrefillSnapshot: Memory.Snapshot

    init(
      cache: [any KVCache],
      outputState: LMOutput.State?,
      sampler: any LogitSampler,
      processor: (any LogitProcessor)?,
      parameters: EdgeToolsMLXGenerateParameters,
      generationStartSnapshot: Memory.Snapshot,
      postPrefillSnapshot: Memory.Snapshot
    ) {
      self.cache = cache
      self.outputState = outputState
      self.sampler = sampler
      self.processor = processor
      self.kvCacheQuantizationBits = parameters.kvCacheQuantizationBits
      self.kvCacheQuantizationGroupSize = parameters.kvCacheQuantizationGroupSize
      self.quantizedKVStart = parameters.quantizedKVStart
      self.synchronizeStreamForMemorySnapshots =
        parameters.synchronizeStreamForMemorySnapshots
      self.generationStartSnapshot = generationStartSnapshot
      self.postPrefillSnapshot = postPrefillSnapshot
    }
  }

  // MARK: - MLX Language Model Defaults

  extension EdgeToolsMLXModel {
    public func tokenIds(in input: LMInput) -> [EdgeToolsToken.ID] {
      input.text.tokens.asArray(EdgeToolsToken.ID.self)
    }

    public nonisolated(nonsending) func prepare(
      input: LMInput,
      parameters: EdgeToolsMLXGenerateParameters,
      assets: EdgeToolsMLXModelAssets
    ) async throws -> EdgeToolsModelPreparation<MLXArray, EdgeToolsMLXGenerationState> {
      let clock = ContinuousClock()
      let generationStartSnapshot = Self.memorySnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let tokenIds = self.tokenIds(in: input)
      let start = clock.now
      var processor = parameters.processor
      processor?.prompt(input.text.tokens)
      let prepared = try self.preparedOutput(
        input: input,
        tokenIds: tokenIds,
        assets: assets
      )
      let metrics = EdgeToolsPrefillMetrics(
        tokens: prepared.tokenCount,
        duration: start.duration(to: clock.now)
      )
      let postPrefillSnapshot = Self.memorySnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let state = EdgeToolsMLXGenerationState(
        cache: prepared.cache,
        outputState: prepared.output.state,
        sampler: parameters.sampler,
        processor: processor,
        parameters: parameters,
        generationStartSnapshot: generationStartSnapshot,
        postPrefillSnapshot: postPrefillSnapshot
      )
      return EdgeToolsModelPreparation(
        logits: prepared.output.logits,
        state: state,
        metrics: metrics
      )
    }

    public nonisolated(nonsending) func decode(
      tokenId: EdgeToolsToken.ID,
      state: inout EdgeToolsMLXGenerationState,
      assets: EdgeToolsMLXModelAssets
    ) async throws -> MLXArray {
      maybeQuantizeKVCache(
        cache: &state.cache,
        kvBits: state.kvCacheQuantizationBits,
        kvGroupSize: state.kvCacheQuantizationGroupSize,
        quantizedKVStart: state.quantizedKVStart
      )
      let token = MLXArray([tokenId])
      let output = self(
        LMInput.Text(tokens: token)[text: .newAxis],
        cache: state.cache,
        state: state.outputState
      )
      state.outputState = output.state
      return output.logits
    }

    public nonisolated(nonsending) func sample(
      logits: inout MLXArray,
      bitmask: GrammarBitmask,
      state: inout EdgeToolsMLXGenerationState
    ) async throws -> EdgeToolsModelSample {
      var stepLogits = logits[0..., -1, 0...]
      stepLogits = state.processor?.process(logits: stepLogits) ?? stepLogits
      let maskedLogits = applyBitmaskMLX(logits: stepLogits, mask: bitmask)
      let confidence = tokenConfidenceMLX(logits: maskedLogits)
      let tokenId = state.sampler.sample(logits: maskedLogits).item(EdgeToolsToken.ID.self)
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public func didAccept(
      token: EdgeToolsToken,
      state: inout EdgeToolsMLXGenerationState
    ) {
      state.processor?.didSample(token: MLXArray([token.id]))
    }

    public func finish(state: EdgeToolsMLXGenerationState) -> EdgeToolsMetadata {
      var metadata = EdgeToolsMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = state.generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = state.postPrefillSnapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = Self.memorySnapshot(
        synchronize: state.synchronizeStreamForMemorySnapshots
      )
      return metadata
    }

    public nonisolated(nonsending) func prefill(
      input: LMInput,
      assets: EdgeToolsMLXModelAssets
    ) async throws -> EdgeToolsEnginePrefill {
      let clock = ContinuousClock()
      let tokenIds = self.tokenIds(in: input)
      let cache = self.newCache(parameters: nil)
      let start = clock.now
      let output = try self.edgeToolsPrepare(input: input, cache: cache)
      eval(output.logits)
      eval(cache)
      assets.cachedPrefill = EdgeToolsMLXModelAssets.CachedPrefill(
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
      tokenIds: [EdgeToolsToken.ID],
      assets: EdgeToolsMLXModelAssets
    ) throws -> (output: LMOutput, cache: [any KVCache], tokenCount: Int) {
      guard let cachedPrefill = assets.cachedPrefill,
        tokenIds.starts(with: cachedPrefill.tokenIds)
      else {
        let cache = self.newCache(parameters: nil)
        return (
          try self.edgeToolsPrepare(input: input, cache: cache),
          cache,
          tokenIds.count
        )
      }
      let suffixCount = tokenIds.count - cachedPrefill.tokenIds.count
      let cache = cachedPrefill.cache.map { $0.copy() }
      let output =
        if suffixCount == 0 {
          cachedPrefill.output
        } else {
          self(
            input.text[cachedPrefill.tokenIds.count...][text: .newAxis],
            cache: cache,
            state: cachedPrefill.output.state
          )
        }
      return (output, cache, suffixCount)
    }

    private func edgeToolsPrepare(
      input: LMInput,
      cache: [any KVCache]
    ) throws -> LMOutput {
      switch try self.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw EdgeToolsMLXError(code: .emptyInput, message: "Model received empty input.")
        }
        return self(
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

  // MARK: - MLX Model Loading

  extension EdgeToolsModelEngine where Model: EdgeToolsMLXModel {
    public init(
      from directoryURL: URL,
      model makeModel: (Model.ModelConfiguration) throws -> Model
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: directoryURL)
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let model = try Model.loadEdgeToolsLanguageModel(
        from: directoryURL,
        model: makeModel
      )
      try self.init(
        model: model,
        assets: EdgeToolsMLXModelAssets(),
        tokenizer: tokenizer
      )
    }
  }

  // MARK: - Prompt Conversion

  #if Transformers
    extension EdgeToolsModel
    where Self: LLMModel, Prompt == EdgeToolsLLMPrompt, Input == LMInput {
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
#endif
