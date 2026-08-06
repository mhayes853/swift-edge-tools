#if XGrammar
  import EdgeToolsXGrammar
#endif

#if Atomics
  import Atomics

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
    associatedtype Tokenizer: EdgeToolsTokenizer
    associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
    associatedtype ToolCallParser: EdgeToolCallParser
    associatedtype GrammarContext = Void
    associatedtype GrammarCompiler: EdgeToolsGrammarCompiler, ~Copyable
    where GrammarCompiler.Context == GrammarContext

    var vocabularySize: Int { get }

    func grammarContext(tokenizer: Tokenizer) throws -> GrammarContext
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

    func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: Tokenizer
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

  extension EdgeToolsModel where GrammarContext == Void {
    public func grammarContext(tokenizer _: Tokenizer) throws {}
  }

  extension EdgeToolsModel {
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

    private let generationGate = EdgeToolsModelGenerationGate()
    private var grammarCompiler: Model.GrammarCompiler
    private let grammarContext: Model.GrammarContext
    private var model: Model
    private let tokenizer: Model.Tokenizer
    private let clock = ContinuousClock()

    public init(
      model: sending Model,
      tokenizer: sending Model.Tokenizer
    ) throws {
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
      let input = try self.model.input(prompt: prompt, tools: tools, tokenizer: self.tokenizer)
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
      // swiftlint:disable:next identifier_name
      model: inout Model
    ) async throws -> EdgeToolsEngineGeneration {
      try Task.checkCancellation()
      guard !isStopped.load(ordering: .relaxed) else { return .empty }

      let grammar = try model.grammar(
        tools: tools,
        parameters: parameters,
        context: self.grammarContext
      )
      var matcher = try self.matcher(grammar: grammar)
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
        generatedTokens.last?.id != self.tokenizer.eosTokenId {
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
          generatedTokens.last?.id != self.tokenizer.eosTokenId {
          try Task.checkCancellation()
          bitmask = matcher.bitmask()
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
      grammar: Model.GrammarCompiler.Grammar
    ) throws -> Model.GrammarCompiler.Matcher {
      try self.grammarCompiler.matcher(for: grammar, context: self.grammarContext)
    }
  }

  #if XGrammar
    extension EdgeToolsModelEngine
    where Model.Tokenizer == AnyEdgeToolsXGRTokenizer {
      public init(
        model: sending Model,
        tokenizer: sending any EdgeToolsXGRTokenizer
      ) throws {
        try self.init(
          model: model,
          tokenizer: AnyEdgeToolsXGRTokenizer(tokenizer)
        )
      }
    }

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
