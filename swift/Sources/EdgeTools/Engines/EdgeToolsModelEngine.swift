#if XGrammar
  import EdgeToolsXGrammar
#endif

#if Atomics
  import Atomics
  import OrderedCollections

  // MARK: - EdgeToolsModelInput

  public struct EdgeToolsModelInput<Value> {
    public var value: Value
    public var tokenIds: [EdgeToolsToken.ID]

    public init(value: Value, tokenIds: [EdgeToolsToken.ID]) {
      self.value = value
      self.tokenIds = tokenIds
    }
  }

  extension EdgeToolsModelInput: Equatable where Value: Equatable {}
  extension EdgeToolsModelInput: Hashable where Value: Hashable {}
  extension EdgeToolsModelInput: Sendable where Value: Sendable {}

  // MARK: - EdgeToolsModelPreparation

  public struct EdgeToolsModelPreparation: Sendable {
    public var metrics: EdgeToolsPrefillMetrics
    public var metadata: EdgeToolsMetadata

    public init(
      metrics: EdgeToolsPrefillMetrics,
      metadata: EdgeToolsMetadata = EdgeToolsMetadata()
    ) {
      self.metrics = metrics
      self.metadata = metadata
    }
  }

  // MARK: - EdgeToolsModelSample

  public struct EdgeToolsModelSample: Hashable, Sendable {
    public var tokenId: EdgeToolsToken.ID
    public var confidence: Float

    public init(tokenId: EdgeToolsToken.ID, confidence: Float) {
      self.tokenId = tokenId
      self.confidence = confidence
    }
  }

  // MARK: - EdgeToolsModel

  public protocol EdgeToolsModel: SendableMetatype {
    associatedtype Prompt: Sendable
    associatedtype Input
    associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
    associatedtype ToolCallParser: EdgeToolCallParser
    associatedtype GrammarContext = Void
    associatedtype GrammarCompiler: EdgeToolsGrammarCompiler, ~Copyable
    where GrammarCompiler.Context == GrammarContext

    var vocabularySize: Int { get }
    var extraStopTokenIds: Set<EdgeToolsToken.ID> { get }

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
    ) async throws -> EdgeToolsModelInput<Input>

    nonisolated(nonsending) mutating func prepare(
      input: Input,
      parameters: GenerateParameters
    ) async throws -> EdgeToolsModelPreparation

    nonisolated(nonsending) mutating func decode(
      bitmask: GrammarBitmask,
      parameters: GenerateParameters
    ) async throws -> EdgeToolsModelSample

    func finish() -> EdgeToolsMetadata

    mutating func resetGeneration()
  }

  extension EdgeToolsModel where GrammarContext == Void {
    public func grammarContext(tokenizer _: any EdgeToolsTokenizer) throws {}
  }

  extension EdgeToolsModel {
    public var extraStopTokenIds: Set<EdgeToolsToken.ID> { [] }

    public func finish() -> EdgeToolsMetadata {
      EdgeToolsMetadata()
    }

    public mutating func resetGeneration() {}
  }

  extension EdgeToolsModel
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

  // MARK: - EdgeToolsPrefillableModel

  public protocol EdgeToolsPrefillableModel: EdgeToolsModel {
    nonisolated(nonsending) mutating func prefill(
      input: Input
    ) async throws -> EdgeToolsEnginePrefill
  }

  // MARK: - EdgeToolsModelEngine

  public actor EdgeToolsModelEngine<Model: EdgeToolsModel>: EdgeToolsEngine {
    public typealias Prompt = Model.Prompt
    public typealias GenerateParameters = Model.GenerateParameters

    private let generationGate = ModelGenerationGate()
    private var grammarCompiler: Model.GrammarCompiler
    private let grammarContext: Model.GrammarContext
    private var model: Model
    private let tokenizer: any EdgeToolsTokenizer
    private let clock = ContinuousClock()

    public init(model: sending Model, tokenizer: sending any EdgeToolsTokenizer) throws {
      let grammarContext = try model.grammarContext(tokenizer: tokenizer)
      self.grammarCompiler = try model.grammarCompiler(context: grammarContext)
      self.grammarContext = grammarContext
      self.model = model
      self.tokenizer = tokenizer
    }

    public func tokenize(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      let input = try await self.model.input(
        prompt: prompt,
        tools: tools,
        tokenizer: self.tokenizer
      )
      let tokens = self.tokenizer.convertIdsToTokens(input.tokenIds)
      return zip(input.tokenIds, tokens)
        .compactMap { tokenId, token in
          token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
        }
    }

    public nonisolated func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Model.GenerateParameters,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let state = ManagedAtomic(AtomicGenerationTask.State.queued.rawValue)

      // NB: Compiler region isolation checker limitation, these are safe because neither value
      // is accessed after being sent to the generation task.
      nonisolated(unsafe) let parameters = parameters
      nonisolated(unsafe) let channel = channel
      let task = Task {
        do {
          try await self.generationGate.acquire()
        } catch {
          let currentState = state.load(ordering: .relaxed)
          guard currentState == AtomicGenerationTask.State.stopped.rawValue else { throw error }
          return EdgeToolsEngineGeneration.empty
        }
        let didStart =
          state.compareExchange(
            expected: AtomicGenerationTask.State.queued.rawValue,
            desired: AtomicGenerationTask.State.running.rawValue,
            ordering: .relaxed
          )
          .exchanged
        guard didStart else {
          await self.generationGate.release()
          return EdgeToolsEngineGeneration.empty
        }
        do {
          let generation = try await self.generate(
            prompt: prompt,
            tools: tools,
            parameters: parameters,
            channel: channel,
            state: state
          )
          await self.generationGate.release()
          return generation
        } catch {
          await self.generationGate.release()
          throw error
        }
      }
      return AtomicGenerationTask(task: task, state: state)
    }

    private func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      parameters: sending Model.GenerateParameters,
      channel: sending EdgeToolsGenerationChannel,
      state: ManagedAtomic<UInt8>
    ) async throws -> EdgeToolsEngineGeneration {
      var model = self.model
      do {
        let generation = try await self.runGeneration(
          prompt: prompt,
          tools: tools,
          parameters: parameters,
          channel: channel,
          state: state,
          model: &model
        )
        model.resetGeneration()
        self.model = model
        return generation
      } catch {
        model.resetGeneration()
        self.model = model
        throw error
      }
    }

    private func runGeneration(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      parameters: sending Model.GenerateParameters,
      channel: sending EdgeToolsGenerationChannel,
      state: ManagedAtomic<UInt8>,
      model: inout Model
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      let initialState = state.load(ordering: .relaxed)
      guard initialState != AtomicGenerationTask.State.stopped.rawValue else { return .empty }

      let grammar = try model.grammar(
        tools: tools,
        parameters: parameters,
        context: self.grammarContext
      )
      var stopTokenIds = model.extraStopTokenIds
      if let eosTokenId = self.tokenizer.eosTokenId {
        stopTokenIds.insert(eosTokenId)
      }

      var matcher = try self.matcher(grammar: grammar, stopTokenIds: stopTokenIds)
      let generateStart = self.clock.now
      let input = try await model.input(prompt: prompt, tools: tools, tokenizer: self.tokenizer)
      var preparation = try await model.prepare(input: input.value, parameters: parameters)
      var detokenizer = StreamingDetokenizer()
      var parser = Model.ToolCallParser()
      var generatedTokens = [EdgeToolsToken]()
      var toolCalls = [EdgeRawToolCall]()
      var confidence = ConfidenceState()
      var durationToFirstToken: Duration?
      let maximumTokenCount = parameters.maxTokens ?? .max

      var bitmask: GrammarBitmask?
      let stateBeforeFirstToken = state.load(ordering: .relaxed)
      if !matcher.isTerminated,
        stateBeforeFirstToken != AtomicGenerationTask.State.stopped.rawValue,
        generatedTokens.count < maximumTokenCount,
        generatedTokens.last.map({ !stopTokenIds.contains($0.id) }) ?? true
      {
        try Task.checkCancellation()
        bitmask = matcher.bitmask()
      }

      while let currentBitmask = bitmask {
        let sample = try await model.decode(
          bitmask: currentBitmask,
          parameters: parameters
        )
        durationToFirstToken =
          durationToFirstToken ?? generateStart.duration(to: self.clock.now)
        confidence.add(confidence: sample.confidence)

        let tokenString = detokenizer.decode(tokenId: sample.tokenId, using: self.tokenizer)
        let token = EdgeToolsToken(id: sample.tokenId, stringValue: tokenString)
        generatedTokens.append(token)
        guard matcher.accept(tokenId: token.id) else {
          throw EdgeToolsError.grammarRejectedToken(token: token)
        }

        channel.emit(token: token)
        for rawToolCall in parser.accept(token: token) {
          toolCalls.append(rawToolCall)
          channel.emit(toolCall: rawToolCall)
        }

        bitmask = nil
        let currentState = state.load(ordering: .relaxed)
        if !matcher.isTerminated,
          currentState != AtomicGenerationTask.State.stopped.rawValue,
          generatedTokens.count < maximumTokenCount,
          generatedTokens.last.map({ !stopTokenIds.contains($0.id) }) ?? true
        {
          try Task.checkCancellation()
          bitmask = matcher.bitmask()
        }
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      let finalMetadata = model.finish()
      preparation.metadata.merge(finalMetadata) { _, finalValue in finalValue }
      preparation.metadata.generationConfidence = confidence.mean
      preparation.metadata.perTokenConfidences = confidence.perTokenConfidences
      var responseTokenIds = detokenizer.tokenIds
      if let lastTokenId = responseTokenIds.last, stopTokenIds.contains(lastTokenId) {
        responseTokenIds.removeLast()
      }
      let response = self.tokenizer.decode(tokens: responseTokenIds)
      let decodeDuration =
        generateStart.duration(to: self.clock.now) - finalDurationToFirstToken
      let finalState = state.load(ordering: .relaxed)
      return EdgeToolsEngineGeneration(
        prefillMetrics: preparation.metrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: generatedTokens.count,
          duration: decodeDuration,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: finalState == AtomicGenerationTask.State.stopped.rawValue,
        tokens: generatedTokens,
        response: response,
        toolCalls: toolCalls,
        metadata: preparation.metadata
      )
    }

    private func matcher(
      grammar: Model.GrammarCompiler.Grammar,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> Model.GrammarCompiler.Matcher {
      try self.grammarCompiler.matcher(
        for: grammar,
        context: self.grammarContext,
        stopTokenIds: stopTokenIds
      )
    }
  }

  #if XGrammar
    extension EdgeToolsModelEngine
    where Model.GrammarCompiler == XGRCompiler, Model.GrammarContext == XGRGrammarContext {
      public func clearCaches() {
        self.grammarContext.clearMatcherCache()
        self.grammarCompiler.clearCache()
      }
    }
  #endif

  extension EdgeToolsModelEngine: EdgeToolsPrefillableEngine
  where Model: EdgeToolsPrefillableModel {
    public func prefill(
      promptPrefix: Model.Prompt,
      tools: [EdgeToolDefinition]
    ) async throws -> EdgeToolsEnginePrefill {
      try await self.generationGate.acquire()
      var model = self.model
      do {
        let input = try await model.input(
          prompt: promptPrefix,
          tools: tools,
          tokenizer: self.tokenizer
        )
        let prefill = try await model.prefill(input: input.value)
        self.model = model
        await self.generationGate.release()
        return prefill
      } catch {
        model.resetGeneration()
        self.model = model
        await self.generationGate.release()
        throw error
      }
    }
  }

  // MARK: - ModelGenerationGate

  private actor ModelGenerationGate {
    private var isAcquired = false
    private var waiters = OrderedDictionary<Int, UnsafeContinuation<Bool, Never>>()
    private var nextID = 0

    func acquire() async throws {
      try Task.checkCancellation()
      guard self.isAcquired else {
        self.isAcquired = true
        return
      }

      let id = self.nextID
      self.nextID += 1
      let wasAcquired = await withTaskCancellationHandler {
        await withUnsafeContinuation { continuation in
          guard !Task.isCancelled else {
            continuation.resume(returning: false)
            return
          }
          self.waiters[id] = continuation
        }
      } onCancel: {
        Task { await self.cancelWaiter(id: id) }
      }
      guard wasAcquired else { throw CancellationError() }
    }

    func release() {
      guard let id = self.waiters.keys.first else {
        self.isAcquired = false
        return
      }
      self.waiters.removeValue(forKey: id)?.resume(returning: true)
    }

    private func cancelWaiter(id: Int) {
      self.waiters.removeValue(forKey: id)?.resume(returning: false)
    }
  }
#endif
