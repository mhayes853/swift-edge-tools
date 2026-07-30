#if XGrammar
  import EdgeToolsXGrammar
#endif

#if XGrammar && Atomics
  import Atomics

  // MARK: - EdgeToolsModelEngineGenerateParameters

  public protocol EdgeToolsModelEngineGenerateParameters: EdgeToolsEngineGenerateParameters {
    var constraint: EdgeToolsXGRGenerationConstraint { get }
    var maxTokens: Int? { get }
  }

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

  /// A model that drives a single generation at a time for ``EdgeToolsModelEngine``.
  ///
  /// A conformance owns both its global assets and the transient state of the generation that is
  /// currently in flight. ``prepare(input:parameters:)`` starts a generation, each
  /// ``decode(bitmask:parameters:)`` produces exactly one token, and ``resetGeneration()`` releases the
  /// transient state once the generation succeeds, fails, or is cancelled.
  public protocol EdgeToolsModel: SendableMetatype {
    associatedtype Prompt: Sendable
    associatedtype Input
    associatedtype GenerateParameters: EdgeToolsModelEngineGenerateParameters
    associatedtype ToolCallParser: EdgeToolCallParser

    var vocabularySize: Int { get }

    func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar

    func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> EdgeToolsModelInput<Input>

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

  extension EdgeToolsModel {
    public func finish() -> EdgeToolsMetadata {
      EdgeToolsMetadata()
    }

    public mutating func resetGeneration() {}
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

    private let generationGate = EdgeToolsModelGenerationGate()
    private let grammarCompiler: XGRCompiler
    private let matcherPool = XGRToolCallMatcherPool()
    private var model: Model
    private let tokenizer: any EdgeToolsXGRTokenizer
    private let clock = ContinuousClock()

    public init(
      model: sending Model,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      self.grammarCompiler = try XGRCompiler(
        tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: model.vocabularySize)
      )
      self.model = model
      self.tokenizer = tokenizer
    }

    public func tokenize(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      let input = try self.model.input(prompt: prompt, tools: tools, tokenizer: self.tokenizer)
      let tokens = self.tokenizer.convertIdsToTokens(input.tokenIds)
      return zip(input.tokenIds, tokens)
        .compactMap { tokenId, token in
          token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
        }
    }

    public func clearCaches() {
      self.matcherPool.clear()
      self.grammarCompiler.clearCache()
    }

    public nonisolated func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Model.GenerateParameters,
      channel: EdgeToolsGenerationChannel
    ) throws -> some EdgeToolsEngineGenerationTask {
      let isStopped = ManagedAtomic(false)

      // NB: Compiler region isolation checker limitation, this is safe because params are not
      // accessed after being sent to generate.
      nonisolated(unsafe) let parameters = parameters
      let task = Task {
        await self.generationGate.acquire()
        do {
          let generation = try await self.generate(
            prompt: prompt,
            tools: tools,
            parameters: parameters,
            channel: channel,
            isStopped: isStopped
          )
          await self.generationGate.release()
          return generation
        } catch {
          await self.generationGate.release()
          throw error
        }
      }
      return AtomicGenerationTask(task: task, isStopped: isStopped)
    }

    private func generate(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition],
      parameters: sending Model.GenerateParameters,
      channel: EdgeToolsGenerationChannel,
      isStopped: ManagedAtomic<Bool>
    ) async throws -> EdgeToolsEngineGeneration {
      var model = self.model
      do {
        let generation = try await self.runGeneration(
          prompt: prompt,
          tools: tools,
          parameters: parameters,
          channel: channel,
          isStopped: isStopped,
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
      channel: EdgeToolsGenerationChannel,
      isStopped: ManagedAtomic<Bool>,
      model: inout Model
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = try self.matcher(tools: tools, constraint: parameters.constraint)
      let generateStart = self.clock.now
      let input = try model.input(prompt: prompt, tools: tools, tokenizer: self.tokenizer)
      var preparation = try await model.prepare(input: input.value, parameters: parameters)
      var detokenizer = StreamingDetokenizer()
      var parser = Model.ToolCallParser()
      var generatedTokens = [EdgeToolsToken]()
      var toolCalls = [EdgeRawToolCall]()
      var confidence = EdgeToolsConfidenceState()
      var durationToFirstToken: Duration?
      let maximumTokenCount = parameters.maxTokens ?? .max

      var bitmask: GrammarBitmask?
      if !matcher.isTerminated,
        !isStopped.load(ordering: .relaxed),
        generatedTokens.count < maximumTokenCount,
        generatedTokens.last?.id != self.tokenizer.eosTokenId
      {
        try Task.checkCancellation()
        bitmask = matcher.grammarBitmask()
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

        let rawToolCall = parser.accept(token: token)
        channel.emit(token: token)
        if let rawToolCall {
          toolCalls.append(rawToolCall)
          channel.emit(toolCall: rawToolCall)
        }

        bitmask = nil
        if !matcher.isTerminated,
          !isStopped.load(ordering: .relaxed),
          generatedTokens.count < maximumTokenCount,
          generatedTokens.last?.id != self.tokenizer.eosTokenId
        {
          try Task.checkCancellation()
          bitmask = matcher.grammarBitmask()
        }
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      let finalMetadata = model.finish()
      preparation.metadata.merge(finalMetadata) { _, finalValue in finalValue }
      preparation.metadata.generationConfidence = confidence.mean
      preparation.metadata.perTokenConfidences = confidence.perTokenConfidences
      let responseTokenIds =
        self.tokenizer.eosTokenId.map { eosTokenId in
          detokenizer.tokenIds.filter { $0 != eosTokenId }
        } ?? detokenizer.tokenIds
      let response = self.tokenizer.decode(tokens: responseTokenIds)
      let decodeDuration =
        generateStart.duration(to: self.clock.now) - finalDurationToFirstToken
      return EdgeToolsEngineGeneration(
        prefillMetrics: preparation.metrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: generatedTokens.count,
          duration: decodeDuration,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: isStopped.load(ordering: .relaxed),
        tokens: generatedTokens,
        response: response,
        toolCalls: toolCalls,
        metadata: preparation.metadata
      )
    }

    private func matcher(
      tools: [EdgeToolDefinition],
      constraint: EdgeToolsXGRGenerationConstraint
    ) throws -> XGRMatcher {
      let toolsGrammar =
        try self.toolCallRange(for: constraint)
        .map { try self.model.grammar(tools: tools, range: $0) } ?? .universal
      let grammar = try self.grammar(for: constraint, toolsGrammar: toolsGrammar)
      let matcher = try self.matcherPool.matcher(
        grammar: grammar,
        compilingWith: self.grammarCompiler
      )
      matcher.reset()
      return matcher
    }

    private func toolCallRange(
      for constraint: EdgeToolsXGRGenerationConstraint
    ) -> GrammarToolCallRange? {
      switch constraint.kind {
      case .toolsWithGrammar(let range, _): range
      default: nil
      }
    }

    private func grammar(
      for constraint: EdgeToolsXGRGenerationConstraint,
      toolsGrammar: XGRGrammar
    ) throws -> XGRGrammar {
      switch constraint.kind {
      case .unconstrained:
        .universal
      case .grammar(let grammar):
        grammar
      case .toolsWithGrammar(_, let transform):
        try transform?(toolsGrammar, self.grammarCompiler.tokenizerInfo) ?? toolsGrammar
      }
    }
  }

  extension EdgeToolsModelEngine: EdgeToolsPrefillableEngine
  where Model: EdgeToolsPrefillableModel {
    public func prefill(
      promptPrefix: Model.Prompt,
      tools: [EdgeToolDefinition]
    ) async throws -> EdgeToolsEnginePrefill {
      await self.generationGate.acquire()
      var model = self.model
      do {
        let input = try model.input(
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

  // MARK: - EdgeToolsModelGenerationGate

  private actor EdgeToolsModelGenerationGate {
    private var isAcquired = false
    private var waiters = [UnsafeContinuation<Void, Never>]()

    func acquire() async {
      guard self.isAcquired else {
        self.isAcquired = true
        return
      }
      await withUnsafeContinuation { self.waiters.append($0) }
    }

    func release() {
      guard !self.waiters.isEmpty else {
        self.isAcquired = false
        return
      }
      self.waiters.removeFirst().resume()
    }
  }
#endif
