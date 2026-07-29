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

  // MARK: - EdgeToolsModelPreparation

  public struct EdgeToolsModelPreparation<Logits, GenerationState> {
    public var logits: Logits
    public var state: GenerationState
    public var metrics: EdgeToolsPrefillMetrics
    public var metadata: EdgeToolsMetadata

    public init(
      logits: Logits,
      state: GenerationState,
      metrics: EdgeToolsPrefillMetrics,
      metadata: EdgeToolsMetadata = EdgeToolsMetadata()
    ) {
      self.logits = logits
      self.state = state
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
    associatedtype Logits
    associatedtype GenerationState
    associatedtype Assets
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
    ) throws -> Input

    func tokenIds(in input: Input) -> [EdgeToolsToken.ID]

    nonisolated(nonsending) func prepare(
      input: Input,
      parameters: GenerateParameters,
      assets: Assets
    ) async throws -> EdgeToolsModelPreparation<Logits, GenerationState>

    nonisolated(nonsending) func decode(
      tokenId: EdgeToolsToken.ID,
      state: inout GenerationState,
      assets: Assets
    ) async throws -> Logits

    nonisolated(nonsending) func sample(
      logits: inout Logits,
      bitmask: GrammarBitmask,
      state: inout GenerationState
    ) async throws -> EdgeToolsModelSample

    func didAccept(
      token: EdgeToolsToken,
      state: inout GenerationState
    )

    func finish(state: GenerationState) -> EdgeToolsMetadata
  }

  extension EdgeToolsModel {
    public func finish(state: GenerationState) -> EdgeToolsMetadata {
      EdgeToolsMetadata()
    }
  }

  // MARK: - EdgeToolsPrefillableModel

  public protocol EdgeToolsPrefillableModel: EdgeToolsModel {
    nonisolated(nonsending) func prefill(
      input: Input,
      assets: Assets
    ) async throws -> EdgeToolsEnginePrefill
  }

  // MARK: - EdgeToolsModelEngine

  public actor EdgeToolsModelEngine<Model: EdgeToolsModel>: EdgeToolsEngine {
    public typealias Prompt = Model.Prompt
    public typealias GenerateParameters = Model.GenerateParameters

    private let generationGate = EdgeToolsModelGenerationGate()
    private let grammarCompiler: XGRCompiler
    private let matcherPool = XGRToolCallMatcherPool()
    private let model: Model
    private let assets: Model.Assets
    private let tokenizer: any EdgeToolsXGRTokenizer
    private let clock = ContinuousClock()

    public init(
      model: sending Model,
      assets: sending Model.Assets,
      tokenizer: sending any EdgeToolsXGRTokenizer
    ) throws {
      self.grammarCompiler = try XGRCompiler(
        tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: model.vocabularySize)
      )
      self.model = model
      self.assets = assets
      self.tokenizer = tokenizer
    }

    public func tokenize(
      prompt: Model.Prompt,
      tools: [EdgeToolDefinition] = []
    ) async throws -> [EdgeToolsToken] {
      let input = try self.model.input(prompt: prompt, tools: tools, tokenizer: self.tokenizer)
      let tokenIds = self.model.tokenIds(in: input)
      let tokens = self.tokenizer.convertIdsToTokens(tokenIds)
      return zip(tokenIds, tokens)
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
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let matcher = try self.matcher(tools: tools, constraint: parameters.constraint)
      let generateStart = self.clock.now
      let input = try self.model.input(prompt: prompt, tools: tools, tokenizer: self.tokenizer)
      var preparation = try await self.model.prepare(
        input: input,
        parameters: parameters,
        assets: self.assets
      )
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
        let sample = try await self.model.sample(
          logits: &preparation.logits,
          bitmask: currentBitmask,
          state: &preparation.state
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
        self.model.didAccept(token: token, state: &preparation.state)

        bitmask = nil
        if !matcher.isTerminated,
          !isStopped.load(ordering: .relaxed),
          generatedTokens.count < maximumTokenCount,
          generatedTokens.last?.id != self.tokenizer.eosTokenId
        {
          try Task.checkCancellation()
          bitmask = matcher.grammarBitmask()
        }
        guard bitmask != nil else { break }
        preparation.logits = try await self.model.decode(
          tokenId: token.id,
          state: &preparation.state,
          assets: self.assets
        )
      }

      let finalDurationToFirstToken = durationToFirstToken ?? .zero
      let finalMetadata = self.model.finish(state: preparation.state)
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
      let toolsGrammar = try self.toolCallRange(for: constraint)
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
      switch constraint {
      case .toolsWithGrammar(let range, _): range
      default: nil
      }
    }

    private func grammar(
      for constraint: EdgeToolsXGRGenerationConstraint,
      toolsGrammar: XGRGrammar
    ) throws -> XGRGrammar {
      switch constraint {
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
      do {
        let input = try self.model.input(
          prompt: promptPrefix,
          tools: tools,
          tokenizer: self.tokenizer
        )
        let prefill = try await self.model.prefill(input: input, assets: self.assets)
        await self.generationGate.release()
        return prefill
      } catch {
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
