#if XGrammar
  import EdgeToolsXGrammar
#endif

#if System
  import SystemPackage
#endif

#if MLX && canImport(MLX)
  import MLX
  import MLXNN
  import MLXLLM
  import MLXLMCommon
  import Foundation
  import Atomics
#endif

#if MLX && Transformers && canImport(MLX)
  import Tokenizers
#endif

// MARK: - Language Model

#if MLX && canImport(MLX)
  public protocol EdgeToolsLanguageModel: LanguageModel {
    associatedtype ModelConfiguration: Decodable
    associatedtype Prompt: Sendable
    associatedtype ToolCallParser: EdgeToolCallParser

    var vocabularySize: Int { get }

    func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar

    func grammarCompiler(
      using tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> XGRCompiler

    func process(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> sending LMInput
  }

  extension EdgeToolsLanguageModel {
    public func grammarCompiler(
      using tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> XGRCompiler {
      try XGRCompiler(
        tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: self.vocabularySize)
      )
    }
  }
#endif

// MARK: - Language Model Loading

#if MLX && canImport(MLX)
  package func loadEdgeToolsLanguageModel<Model: EdgeToolsLanguageModel>(
    _: Model.Type,
    from directoryURL: URL,
    model: (Model.ModelConfiguration) throws -> Model
  ) throws -> Model {
    let configuration = try decodeModelConfiguration(
      Model.ModelConfiguration.self,
      in: directoryURL
    )
    let baseConfiguration = try decodeModelConfiguration(
      BaseConfiguration.self,
      in: directoryURL
    )
    guard let configuration, let baseConfiguration else {
      throw EdgeToolsError.failedToLoadConfiguration
    }

    let model = try model(configuration)
    try loadWeights(
      modelDirectory: directoryURL,
      model: model,
      perLayerQuantization: baseConfiguration.perLayerQuantization
    )
    return model
  }
#endif

// MARK: - MLX Engine

#if MLX && canImport(MLX)
  // MARK: - EdgeToolsMLXEngine

  public final class EdgeToolsMLXEngine<Model: EdgeToolsLanguageModel>: EdgeToolsEngine {
    public typealias Prompt = Model.Prompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self {
        Self()
      }

      private var _sampler: @Sendable () -> any LogitSampler
      private var _processor: @Sendable () -> (any LogitProcessor)?

      public var sampler: any LogitSampler {
        self._sampler()
      }

      public var processor: (any LogitProcessor)? {
        self._processor()
      }

      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any LogitSampler = ArgMaxSampler(),
        processor: @autoclosure @escaping @Sendable () -> (any LogitProcessor)? = nil,
        constraint: XGRGenerationConstraint = .tools,
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self._sampler = sampler
        self._processor = processor
        self.constraint = constraint
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
        self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
      }
    }

    private struct CachedPrefill {
      let tokenIDs: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
    }

    private struct GenerationPreparation {
      let output: LMOutput
      let cache: [any KVCache]
      let metrics: EdgeToolsPrefillMetrics
      let snapshot: Memory.Snapshot
    }

    private struct State: ~Copyable {
      let grammarEngine: XGRCompiler
      let model: Model
      let matcherPool: XGRToolCallMatcherPool
      var cachedPrefill: CachedPrefill?
    }

    private let state: Lock<State>
    private let tokenizer: any EdgeToolsXGRTokenizer
    private let clock = ContinuousClock()

    public init(
      model: sending Model,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      self.state = try Lock {
        let grammarEngine = try model.grammarCompiler(using: tokenizer)
        return State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: XGRToolCallMatcherPool(),
          cachedPrefill: nil
        )
      }
      self.tokenizer = tokenizer
    }

    private init(
      loadingFrom directoryURL: URL,
      tokenizer: sending any EdgeToolsXGRTokenizer,
      model: (Model.ModelConfiguration) throws -> Model
    ) throws {
      self.state = try Lock {
        let model = try loadEdgeToolsLanguageModel(Model.self, from: directoryURL, model: model)
        let grammarEngine = try model.grammarCompiler(using: tokenizer)
        return State(
          grammarEngine: consume grammarEngine,
          model: model,
          matcherPool: XGRToolCallMatcherPool(),
          cachedPrefill: nil
        )
      }
      self.tokenizer = tokenizer
    }

    public convenience init(
      from directoryURL: URL,
      model: (Model.ModelConfiguration) throws -> Model
    ) async throws {
      let tokenizer = try await loadEdgeToolsTokenizer(from: directoryURL)
      guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      try self.init(
        loadingFrom: directoryURL,
        tokenizer: tokenizer,
        model: model
      )
    }

    #if System
      public convenience init(
        from directoryPath: FilePath,
        model: (Model.ModelConfiguration) throws -> Model
      ) async throws {
        try await self.init(
          from: URL(filePath: directoryPath.string, directoryHint: .isDirectory),
          model: model
        )
      }
    #endif

    public func tokenize(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      try self.state.withBorrowedLock { state in
        let input = try state.model.process(
          prompt: prompt,
          tools: tools,
          using: self.tokenizer
        )
        let tokenIDs = input.text.tokens.asArray(EdgeToolsToken.ID.self)
        let tokens = self.tokenizer.convertIdsToTokens(tokenIDs)
        return zip(tokenIDs, tokens)
          .compactMap { tokenID, token in
            token.map { EdgeToolsToken(id: tokenID, stringValue: $0) }
          }
      }
    }

    public func clearCaches() {
      self.state.withLock {
        $0.cachedPrefill = nil
        $0.matcherPool.clear()
        $0.grammarEngine.clearCache()
      }
    }

    public func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        try self.state.withLock { state in
          try self.generate(
            prompt: prompt,
            tools: tools,
            parameters: parameters,
            channel: channel,
            state: &state,
            isStopped: isStopped
          )
        }
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel,
      state: inout sending State,
      isStopped: ManagedAtomic<Bool>
    ) throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let toolsGrammar = try parameters.constraint.toolCallRange.map {
        try state.model.grammar(tools: tools, range: $0)
      } ?? .universal
      let grammar = try parameters.constraint.grammar(
        using: toolsGrammar,
        tokenizerInfo: state.grammarEngine.tokenizerInfo
      )
      let matcher = try state.matcherPool.matcher(
        grammar: grammar,
        compilingWith: state.grammarEngine
      )
      matcher.reset()

      let generationStartSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let generateStart = self.clock.now
      let input = try state.model.process(
        prompt: prompt,
        tools: tools,
        using: self.tokenizer
      )
      let sampler = parameters.sampler
      var processor = parameters.processor
      let preparation = try self.prepareForGeneration(
        input: input,
        state: &state,
        processor: &processor,
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      var output = preparation.output
      var cache = preparation.cache
      try Task.checkCancellation()

      var loop = EdgeToolsGenerationLoop<Model.ToolCallParser>(
        matcher: consume matcher,
        tokenizer: self.tokenizer,
        channel: channel,
        isStopped: isStopped,
        maximumTokenCount: parameters.maxTokens,
        generateStart: generateStart
      )
      var bitmask = try loop.nextBitmask()
      while let currentBitmask = bitmask {
        let processedLogits = processor?.process(logits: output.logits) ?? output.logits
        let logits = applyBitmaskMLX(
          logits: processedLogits[0..., -1, 0...],
          mask: currentBitmask
        )
        let confidence = tokenConfidenceMLX(logits: logits)
        let sampledToken = sampler.sample(logits: logits)
        let tokenID = sampledToken.item(EdgeToolsToken.ID.self)
        _ = try loop.accept(tokenID: tokenID, confidence: confidence)
        processor?.didSample(token: sampledToken)

        bitmask = try loop.nextBitmask()
        guard bitmask != nil else { break }
        maybeQuantizeKVCache(
          cache: &cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize
        )

        let inputText = LMInput.Text(tokens: sampledToken)
        output = state.model(
          inputText[text: .newAxis],
          cache: cache,
          state: output.state
        )
      }

      let postDecodeSnapshot = Memory.synchronizedSnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      var metadata = EdgeToolsMetadata()
      metadata.mlxEngineGenerationStartMemorySnapshot = generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = preparation.snapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = postDecodeSnapshot
      return loop.finish(prefillMetrics: preparation.metrics, metadata: metadata)
    }

    private func prepareForGeneration(
      input: LMInput,
      state: inout State,
      processor: inout (any LogitProcessor)?,
      synchronize: Bool
    ) throws -> GenerationPreparation {
      let tokenIDs = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      if let cachedPrefill = state.cachedPrefill,
        tokenIDs.starts(with: cachedPrefill.tokenIDs)
      {
        let suffixCount = tokenIDs.count - cachedPrefill.tokenIDs.count
        let prefillStart = self.clock.now
        processor?.prompt(input.text.tokens)
        let cache = cachedPrefill.cache.map { $0.copy() }
        let output =
          if suffixCount == 0 {
            cachedPrefill.output
          } else {
            state.model(
              input.text[cachedPrefill.tokenIDs.count...][text: .newAxis],
              cache: cache,
              state: cachedPrefill.output.state
            )
          }
        let metrics = EdgeToolsPrefillMetrics(
          tokens: suffixCount,
          duration: prefillStart.duration(to: self.clock.now)
        )
        let snapshot = Memory.synchronizedSnapshot(synchronize: synchronize)
        return GenerationPreparation(
          output: output,
          cache: cache,
          metrics: metrics,
          snapshot: snapshot
        )
      }

      let cache = state.model.newCache(parameters: nil)
      let prefillStart = self.clock.now
      let output = try self.prepare(
        input: input,
        model: state.model,
        cache: cache,
        processor: &processor
      )
      let metrics = EdgeToolsPrefillMetrics(
        tokens: tokenIDs.count,
        duration: prefillStart.duration(to: self.clock.now)
      )
      let snapshot = Memory.synchronizedSnapshot(synchronize: synchronize)
      return GenerationPreparation(
        output: output,
        cache: cache,
        metrics: metrics,
        snapshot: snapshot
      )
    }

    private func prepare(
      input: LMInput,
      model: Model,
      cache: [any KVCache],
      processor: inout (any LogitProcessor)?
    ) throws -> LMOutput {
      processor?.prompt(input.text.tokens)
      switch try model.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw EdgeToolsMLXError(code: .emptyInput, message: "Model received empty input.")
        }
        return model(
          tokens[text: .newAxis],
          cache: cache.isEmpty ? nil : cache,
          state: nil
        )
      }
    }
  }

  // MARK: - EdgeToolsPrefillableEngine

  extension EdgeToolsMLXEngine: EdgeToolsPrefillableEngine where Model: LLMModel {
    public func prefill(
      promptPrefix: Model.Prompt,
      tools: [EdgeToolDefinition]
    ) async throws -> EdgeToolsEnginePrefill {
      try self.state.withLock { state in
        try Task.checkCancellation()
        let input = try state.model.process(
          prompt: promptPrefix,
          tools: tools,
          using: self.tokenizer
        )
        return try self.prefill(input: input, state: &state)
      }
    }

    private func prefill(
      input: consuming sending LMInput,
      state: inout sending State
    ) throws -> EdgeToolsEnginePrefill {
      try Task.checkCancellation()
      let tokenIDs = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let cache = state.model.newCache(parameters: nil)
      var processor: (any LogitProcessor)?
      let prefillStart = self.clock.now
      let output = try self.prepare(
        input: input,
        model: state.model,
        cache: cache,
        processor: &processor
      )
      eval(output.logits)
      eval(cache)
      state.cachedPrefill = CachedPrefill(
        tokenIDs: tokenIDs,
        cache: cache.map { $0.copy() },
        output: output
      )
      let snapshot = Memory.synchronizedSnapshot(synchronize: true)
      var metadata = EdgeToolsMetadata()
      metadata.mlxEnginePostPrefillMemorySnapshot = snapshot
      return EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(
          tokens: tokenIDs.count,
          duration: prefillStart.duration(to: self.clock.now)
        ),
        metadata: metadata
      )
    }
  }

  // MARK: - Synchronized Memory Snapshot

  extension Memory {
    fileprivate static func synchronizedSnapshot(synchronize: Bool) -> Snapshot {
      if synchronize {
        Stream.defaultStream(.defaultDevice()).synchronize()
      }
      return Self.snapshot()
    }
  }

  // MARK: - EdgeToolsMLXError

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
#endif

// MARK: - Prompt Conversion

#if MLX && Transformers && canImport(MLX)
  extension EdgeToolsLanguageModel where Self: LLMModel, Prompt == EdgeToolsLLMPrompt {
    public func process(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> sending LMInput {
      guard let tokenizer = tokenizer as? EdgeToolsPreTrainedTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let preTrainedTokenizer = tokenizer.tokenizer
      let tokenIDs = try preTrainedTokenizer.applyChatTemplate(
        messages: prompt.mlxMessages(),
        tools: tools.mlxToolSpecs,
        additionalContext: nil
      )
      return LMInput(tokens: MLXArray(tokenIDs))
    }
  }

  extension EdgeToolsLLMPrompt {
    fileprivate func mlxMessages() throws -> [MLXLMCommon.Message] {
      try self.messages.map { try $0.mlxMessage() }
    }

  }

  extension EdgeToolsLLMPrompt.Message {
    package func mlxMessage() throws -> MLXLMCommon.Message {
      var message: MLXLMCommon.Message = ["role": self.role.rawValue]
      if let content {
        message["content"] = content
      }
      if let toolResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        message["content"] = String(decoding: try encoder.encode(toolResponse), as: UTF8.self)
      }
      if let toolName {
        message["name"] = toolName
      }
      if let toolCalls, !toolCalls.isEmpty {
        message["tool_calls"] = toolCalls.map(\.mlxToolCall)
      }
      return message
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
