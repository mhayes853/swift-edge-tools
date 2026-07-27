#if ONNXCore
  import Atomics

  #if Foundation
    import Foundation
  #endif

  #if Foundation && System
    import SystemPackage
  #endif

  // MARK: - EdgeToolsONNXModelPreparation

  public struct EdgeToolsONNXModelPreparation<State> {
    public var logits: [Float]
    public var state: State

    public init(logits: [Float], state: State) {
      self.logits = logits
      self.state = state
    }
  }

  // MARK: - EdgeToolsONNXModel

  public protocol EdgeToolsONNXModel<Runtime>: SendableMetatype {
    associatedtype Runtime: EdgeToolsONNXRuntime
    associatedtype ModelConfiguration: Decodable & Sendable
    associatedtype Prompt: Sendable
    associatedtype ToolCallParser: EdgeToolCallParser
    associatedtype GenerationState

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
    ) throws -> [EdgeToolsToken.ID]

    nonisolated(nonsending) func prepare(
      tokenIDs: [EdgeToolsToken.ID],
      using runtime: Runtime
    ) async throws -> EdgeToolsONNXModelPreparation<GenerationState>

    nonisolated(nonsending) func decode(
      tokenID: EdgeToolsToken.ID,
      state: inout GenerationState,
      using runtime: Runtime
    ) async throws -> [Float]
  }

  extension EdgeToolsONNXModel {
    public func grammarCompiler(
      using tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> XGRCompiler {
      try XGRCompiler(
        tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: self.vocabularySize)
      )
    }
  }

  // MARK: - EdgeToolsPrefillableONNXModel

  public protocol EdgeToolsPrefillableONNXModel: EdgeToolsONNXModel {
    nonisolated(nonsending) func prefill(
      tokenIDs: [EdgeToolsToken.ID],
      using runtime: Runtime
    ) async throws -> EdgeToolsONNXModelPreparation<GenerationState>
  }

  extension EdgeToolsPrefillableONNXModel {
    public func prepare(
      tokenIDs: [EdgeToolsToken.ID],
      using runtime: Runtime
    ) async throws -> EdgeToolsONNXModelPreparation<GenerationState> {
      try await self.prefill(tokenIDs: tokenIDs, using: runtime)
    }
  }

  // MARK: - EdgeToolsONNXEngineComponents

  package struct EdgeToolsONNXEngineComponents<Runtime, Model> {
    package let runtime: Runtime
    package let model: Model

    package init(runtime: Runtime, model: Model) {
      self.runtime = runtime
      self.model = model
    }
  }

  // MARK: - EdgeToolsONNXEngine

  public actor EdgeToolsONNXEngine<
    Runtime: EdgeToolsONNXRuntime,
    Model: EdgeToolsONNXModel<Runtime>
  >: EdgeToolsEngine {
    public typealias Prompt = Model.Prompt

    public struct GenerateParameters: EdgeToolsEngineGenerateParameters {
      public static var `default`: Self { Self() }

      private var _sampler: @Sendable () -> any EdgeToolsSampler<[Float]>
      private var _processor:
        @Sendable () -> (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)?

      public var sampler: any EdgeToolsSampler<[Float]> {
        self._sampler()
      }

      public var processor: (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)? {
        self._processor()
      }

      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?

      public init(
        sampler: @autoclosure @escaping @Sendable () -> any EdgeToolsSampler<[Float]> =
          ONNXArgmaxSampler(),
        processor:
          @autoclosure @escaping @Sendable () ->
          (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)? = nil,
        constraint: XGRGenerationConstraint = .tools,
        maxTokens: Int? = 1024
      ) {
        self._sampler = sampler
        self._processor = processor
        self.constraint = constraint
        self.maxTokens = maxTokens
      }
    }

    private let grammarCompiler: XGRCompiler
    private let matcherPool: XGRToolCallMatcherPool
    private let runtime: Runtime
    private let model: Model
    private let tokenizer: any EdgeToolsXGRTokenizer
    private let clock = ContinuousClock()

    public init(
      runtime: sending Runtime,
      model: sending Model,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      let components = EdgeToolsONNXEngineComponents(runtime: runtime, model: model)
      try self.init(components: components, tokenizer: tokenizer)
    }

    package init(
      components: sending EdgeToolsONNXEngineComponents<Runtime, Model>,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      self.grammarCompiler = try components.model.grammarCompiler(using: tokenizer)
      self.matcherPool = XGRToolCallMatcherPool()
      self.runtime = components.runtime
      self.model = components.model
      self.tokenizer = tokenizer
    }

    #if Foundation
      public init(
        from directoryURL: URL,
        runtime: sending Runtime,
        model: (URL, Model.ModelConfiguration, Runtime) async throws -> sending Model
      ) async throws {
        let tokenizer = try await loadEdgeToolsTokenizer(from: directoryURL)
        guard let tokenizer = tokenizer as? any EdgeToolsXGRTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        guard
          let configuration = try decodeModelConfiguration(
            Model.ModelConfiguration.self,
            in: directoryURL
          )
        else {
          throw EdgeToolsError.failedToLoadConfiguration
        }
        let model = try await model(directoryURL, configuration, runtime)
        try self.init(runtime: runtime, model: model, tokenizer: tokenizer)
      }
    #endif

    #if Foundation && System
      public init(
        from directoryPath: FilePath,
        runtime: sending Runtime,
        model: (URL, Model.ModelConfiguration, Runtime) async throws -> sending Model
      ) async throws {
        try await self.init(
          from: URL(filePath: directoryPath.string, directoryHint: .isDirectory),
          runtime: runtime,
          model: model
        )
      }
    #endif

    public func tokenize(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      let tokenIDs = try self.model.process(
        prompt: prompt,
        tools: tools,
        using: self.tokenizer
      )
      let tokens = self.tokenizer.convertIdsToTokens(tokenIDs)
      return zip(tokenIDs, tokens)
        .compactMap { tokenID, token in
          token.map { EdgeToolsToken(id: tokenID, stringValue: $0) }
        }
    }

    public func clearCaches() {
      self.matcherPool.clear()
      self.grammarCompiler.clearCache()
    }

    public nonisolated func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let isStopped = ManagedAtomic(false)
      let task = Task {
        try await self.generate(
          prompt: prompt,
          tools: tools,
          parameters: parameters,
          channel: channel,
          isStopped: isStopped
        )
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      channel: EdgeToolsGenerationChannel,
      isStopped: ManagedAtomic<Bool>
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let toolsGrammar = try parameters.constraint.toolCallRange.map {
        try self.model.grammar(tools: tools, range: $0)
      } ?? .universal
      let grammar = try parameters.constraint.grammar(using: toolsGrammar)
      let matcher = try self.matcherPool.matcher(
        grammar: grammar,
        compilingWith: self.grammarCompiler
      )
      matcher.reset()
      let sampler = parameters.sampler
      var processor = parameters.processor
      let generateStart = self.clock.now
      let (preparation, preparationMetrics) = try await self.prepare(
        prompt: prompt,
        tools: tools,
        processor: &processor
      )
      var generationState = preparation.state
      var logits = preparation.logits
      var nextTokenID: EdgeToolsToken.ID?
      var loop = EdgeToolsGenerationLoop<Model.ToolCallParser>(
        matcher: consume matcher,
        tokenizer: self.tokenizer,
        channel: channel,
        isStopped: isStopped,
        maximumTokenCount: parameters.maxTokens,
        generateStart: generateStart
      )
      while let bitmask = try loop.nextBitmask() {
        if let nextTokenID {
          logits = try await self.model.decode(
            tokenID: nextTokenID,
            state: &generationState,
            using: self.runtime
          )
        }
        guard logits.count == self.model.vocabularySize else {
          throw EdgeToolsONNXError(
            code: .invalidLogitsCount,
            message: "Expected \(self.model.vocabularySize) logits, got \(logits.count)."
          )
        }
        logits = try await processor?.process(logits: &logits) ?? logits
        applyONNXBitmask(logits: &logits, mask: bitmask)
        let confidence = tokenConfidenceONNX(logits: logits)
        let tokenID = try await sampler.sample(logits: logits)
        let token = try loop.accept(tokenID: tokenID, confidence: confidence)
        nextTokenID = tokenID
        processor?.didSample(token: token)
      }

      return loop.finish(prefillMetrics: preparationMetrics)
    }

    private func prepare(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      processor: inout (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)?
    ) async throws -> (
      EdgeToolsONNXModelPreparation<Model.GenerationState>, EdgeToolsPrefillMetrics
    ) {
      let tokenIDs = try self.model.process(
        prompt: prompt,
        tools: tools,
        using: self.tokenizer
      )
      let preparationStart = self.clock.now
      processor?.prompt(tokenIDs)
      let preparation = try await self.model.prepare(tokenIDs: tokenIDs, using: self.runtime)
      return (
        preparation,
        EdgeToolsPrefillMetrics(
          tokens: tokenIDs.count,
          duration: preparationStart.duration(to: self.clock.now)
        )
      )
    }
  }

  // MARK: - EdgeToolsONNXError

  public struct EdgeToolsONNXError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let integerConversionFailure = Self(rawValue: "integer-conversion-failure")
      public static let invalidLogitsCount = Self(rawValue: "invalid-logits-count")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }
#endif
