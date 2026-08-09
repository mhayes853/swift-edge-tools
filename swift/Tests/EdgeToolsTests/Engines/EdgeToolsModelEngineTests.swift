#if XGrammar && Atomics && Foundation
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `EdgeToolsModelEngine tests` {
    @Test
    func `Generates Through The Shared Model Engine`() async throws {
      let tokenizer = try testTokenizer()
      let responseTokenIds = encodedGrammarText("hello", tokenizer: tokenizer)
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let model = TestModel()
      let engine = try EdgeToolsModelEngine(
        model: model,
        tokenizer: tokenizer
      )
      let task = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Prompt"),
        parameters: TestModel.Parameters(tokenIds: responseTokenIds + [eosTokenId]),
        channel: EdgeToolsGenerationChannel()
      )

      let generation = try await task.value

      expectNoDifference(generation.wasStopped, false)
      expectNoDifference(generation.tokens.map(\.id), responseTokenIds + [eosTokenId])
      expectNoDifference(generation.response, tokenizer.decode(tokens: responseTokenIds))
      expectNoDifference(generation.text, tokenizer.decode(tokens: responseTokenIds))
      expectNoDifference(generation.prefillMetrics.tokens > 0, true)
    }

    @Test
    func `Stops At Any Extra Model Stop Token`() async throws {
      let tokenizer = try testTokenizer()
      let responseTokenIds = encodedGrammarText("hello", tokenizer: tokenizer)
      let stopTokenId = try #require(tokenizer.unknownTokenId)
      let alternateStopTokenId = try #require(tokenizer.bosTokenId)
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let engine = try EdgeToolsModelEngine(
        model: TestModel(extraStopTokenIds: [alternateStopTokenId, stopTokenId]),
        tokenizer: tokenizer
      )
      let task = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Prompt"),
        parameters: TestModel.Parameters(
          tokenIds: responseTokenIds + [stopTokenId, eosTokenId]
        ),
        channel: EdgeToolsGenerationChannel()
      )

      let generation = try await task.value

      expectNoDifference(generation.tokens.map(\.id), responseTokenIds + [stopTokenId])
      expectNoDifference(generation.response, tokenizer.decode(tokens: responseTokenIds))
    }

    @Test
    func `Constrained Model Parameters Resolve Tool Constraints`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let observation = ConstraintObservation()
      let model = TestModel(constraintObservation: observation)
      let engine = try EdgeToolsModelEngine(
        model: model,
        tokenizer: tokenizer
      )
      let range = GrammarToolCallRange.exact(1)
      let constraint = XGRGenerationConstraint.toolsWithGrammar(
        range: range,
        grammar: { toolsGrammar, tokenizerInfo in
          observation.recordTransform(tokenizerInfo: tokenizerInfo)
          return toolsGrammar
        }
      )
      let task = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Prompt"),
        parameters: TestModel.Parameters(
          tokenIds: [eosTokenId],
          constraint: constraint
        ),
        channel: EdgeToolsGenerationChannel()
      )

      _ = try await task.value

      expectNoDifference(observation.toolCallRange, range)
      expectNoDifference(observation.didTransform, true)
      expectNoDifference(observation.didReceiveTokenizerInfo, true)
    }

    @Test
    func `Constrained Model Parameters Union Tool And User Grammars`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let observation = ConstraintObservation()
      let model = TestModel(constraintObservation: observation)
      let engine = try EdgeToolsModelEngine(
        model: model,
        tokenizer: tokenizer
      )
      let range = GrammarToolCallRange.exact(1)
      let userGrammar = try XGRGrammar.literal("USER")
      let constraint = XGRGenerationConstraint.toolsOrGrammar(
        userGrammar,
        range: range,
        transform: { toolsGrammar, tokenizerInfo in
          observation.recordTransform(tokenizerInfo: tokenizerInfo)
          return toolsGrammar
        }
      )
      let task = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Prompt"),
        parameters: TestModel.Parameters(
          tokenIds: [eosTokenId],
          constraint: constraint
        ),
        channel: EdgeToolsGenerationChannel()
      )

      _ = try await task.value

      expectNoDifference(observation.toolCallRange, range)
      expectNoDifference(observation.didTransform, true)
      expectNoDifference(observation.didReceiveTokenizerInfo, true)
    }

    @Test
    func `Serializes Concurrent Generations`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let assets = TestAssets()
      let engine = try EdgeToolsModelEngine(
        model: TestModel(assets: assets),
        tokenizer: tokenizer
      )
      let parameters = TestModel.Parameters(
        tokenIds: [eosTokenId],
        preparationDelay: .milliseconds(50)
      )
      let first = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "First"),
        parameters: parameters,
        channel: EdgeToolsGenerationChannel()
      )
      let second = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Second"),
        parameters: parameters,
        channel: EdgeToolsGenerationChannel()
      )

      async let firstGeneration = first.value
      async let secondGeneration = second.value
      _ = try await (firstGeneration, secondGeneration)

      expectNoDifference(assets.maximumActiveCount, 1)
    }

    @Test(.timeLimit(.minutes(1)))
    func `Cancelling A Queued Generation Does Not Wait For The Active Generation`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let assets = TestAssets()
      let preparationGate = TestPreparationGate()
      let engine = try EdgeToolsModelEngine(
        model: TestModel(assets: assets),
        tokenizer: tokenizer
      )
      let first = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "First"),
        parameters: TestModel.Parameters(
          tokenIds: [eosTokenId],
          preparationGate: preparationGate
        ),
        channel: EdgeToolsGenerationChannel()
      )
      let firstValue = Task { try await first.value }
      await preparationGate.waitUntilEntered()
      defer { preparationGate.open() }

      let second = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Second"),
        parameters: TestModel.Parameters(tokenIds: [eosTokenId]),
        channel: EdgeToolsGenerationChannel()
      )
      let secondValue = Task { try await second.value }
      await Task.yield()

      secondValue.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await secondValue.value
      }

      expectNoDifference(preparationGate.isOpen, false)
      expectNoDifference(assets.activeCount, 1)
      preparationGate.open()
      _ = try await firstValue.value
    }

    @Test(.timeLimit(.minutes(1)))
    func `Stopping A Queued Generation Does Not Wait For The Active Generation`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let assets = TestAssets()
      let preparationGate = TestPreparationGate()
      let engine = try EdgeToolsModelEngine(
        model: TestModel(assets: assets),
        tokenizer: tokenizer
      )
      let first = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "First"),
        parameters: TestModel.Parameters(
          tokenIds: [eosTokenId],
          preparationGate: preparationGate
        ),
        channel: EdgeToolsGenerationChannel()
      )
      let firstValue = Task { try await first.value }
      await preparationGate.waitUntilEntered()
      defer { preparationGate.open() }

      let second = try engine.generate(
        prompt: NeedlePrompt(system: "", user: "Second"),
        parameters: TestModel.Parameters(tokenIds: [eosTokenId]),
        channel: EdgeToolsGenerationChannel()
      )
      second.stop()

      let generation = try await second.value

      expectNoDifference(generation.isEmpty, true)
      expectNoDifference(preparationGate.isOpen, false)
      expectNoDifference(assets.activeCount, 1)
      preparationGate.open()
      _ = try await firstValue.value
    }
  }

  private final class TestPreparationGate: Sendable {
    private let entered = AsyncStream<Void>.makeStream()
    private let opened = AsyncStream<Void>.makeStream()
    private let openState = Lock(false)

    var isOpen: Bool {
      self.openState.withLock { $0 }
    }

    func waitUntilEntered() async {
      for await _ in self.entered.stream { return }
    }

    func wait() async {
      self.entered.continuation.yield()
      self.entered.continuation.finish()
      guard !self.isOpen else { return }
      for await _ in self.opened.stream { return }
    }

    func open() {
      let didOpen = self.openState.withLock { isOpen in
        guard !isOpen else { return false }
        isOpen = true
        return true
      }
      guard didOpen else { return }
      self.opened.continuation.yield()
      self.opened.continuation.finish()
    }
  }

  private final class TestAssets: Sendable {
    private struct State: Hashable, Sendable {
      var activeCount = 0
      var maximumActiveCount = 0
    }

    private let state = Lock(State())

    var maximumActiveCount: Int {
      self.state.withLock { $0.maximumActiveCount }
    }

    var activeCount: Int {
      self.state.withLock { $0.activeCount }
    }

    func begin() {
      self.state.withLock { state in
        state.activeCount += 1
        state.maximumActiveCount = max(state.maximumActiveCount, state.activeCount)
      }
    }

    func end() {
      self.state.withLock { $0.activeCount -= 1 }
    }
  }

  private final class ConstraintObservation: Sendable {
    private let range = Lock<GrammarToolCallRange?>(nil)
    private let transformed = Lock(false)
    private let tokenizerInfo = Lock<XGRTokenizerInfo?>(nil)

    var toolCallRange: GrammarToolCallRange? {
      self.range.withLock { $0 }
    }

    var didTransform: Bool {
      self.transformed.withLock { $0 }
    }

    var didReceiveTokenizerInfo: Bool {
      self.tokenizerInfo.withLock { $0 != nil }
    }

    func record(range: GrammarToolCallRange) {
      self.range.withLock { $0 = range }
    }

    func recordTransform(tokenizerInfo: XGRTokenizerInfo) {
      self.transformed.withLock { $0 = true }
      self.tokenizerInfo.withLock { $0 = tokenizerInfo }
    }
  }

  private struct TestModel: EdgeToolsModel, Sendable {
    struct Parameters: EdgeToolsConstrainedGenerateParameters {
      static var `default`: Self { Parameters(tokenIds: []) }

      var tokenIds: [EdgeToolsToken.ID]
      var preparationDelay = Duration.zero
      var preparationGate: TestPreparationGate?
      var constraint = XGRGenerationConstraint.unconstrained
      var maxTokens: Int? = 32
    }

    typealias Prompt = NeedlePrompt
    typealias GenerateParameters = Parameters
    typealias GenerationParser = NeedleGenerationParser
    typealias GrammarContext = XGRGrammarContext

    var assets: TestAssets?
    var constraintObservation: ConstraintObservation?
    var extraStopTokenIds = Set<EdgeToolsToken.ID>()
    var index = 0

    var vocabularySize: Int { .needleVocabularySize }

    func grammarContext(tokenizer: any EdgeToolsTokenizer) throws -> XGRGrammarContext {
      guard let tokenizer = tokenizer as? any XGRTokenizer else {
        throw XGRError(
          code: .invalidTokenizerInfo,
          message: "The test model requires an XGrammar tokenizer."
        )
      }
      return try XGRGrammarContext(
        tokenizerInfo: tokenizer.tokenizerInfo(
          modelVocabularySize: self.vocabularySize,
          extraStopTokenIds: self.extraStopTokenIds
        )
      )
    }

    func grammarCompiler(context: borrowing XGRGrammarContext) throws -> XGRCompiler {
      try XGRCompiler(tokenizerInfo: context.tokenizerInfo)
    }

    func grammar(
      prompt _: NeedlePrompt,
      tools _: [EdgeToolDefinition],
      parameters: Parameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      let constraint = parameters.constraint
      let grammar = try constraint.toolCallRange.map {
        self.constraintObservation?.record(range: $0)
        return XGRGrammar.universal
      }
      return try constraint.grammar(toolCallGrammar: grammar, context: context)
    }

    func tokenIds(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) throws -> [EdgeToolsToken.ID] {
      tokenizer.encode(text: prompt.user)
    }

    nonisolated(nonsending) mutating func prepare(
      prompt: inout NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      parameters: Parameters,
      parser _: inout NeedleGenerationParser
    ) async throws -> EdgeToolsModelPreparation {
      let tokenIds = try self.tokenIds(prompt: prompt, tools: tools, tokenizer: tokenizer)
      self.assets?.begin()
      defer { self.assets?.end() }
      if let preparationGate = parameters.preparationGate {
        await preparationGate.wait()
      } else {
        try await Task.sleep(for: parameters.preparationDelay)
      }
      self.index = 0
      return EdgeToolsModelPreparation(
        metrics: EdgeToolsPrefillMetrics(tokens: tokenIds.count, duration: .zero)
      )
    }

    nonisolated(nonsending) mutating func decode(
      bitmask: GrammarBitmask,
      parameters: Parameters
    ) async throws -> EdgeToolsModelSample {
      let tokenId = parameters.tokenIds[self.index]
      self.index += 1
      return EdgeToolsModelSample(tokenId: tokenId, confidence: 1)
    }

    mutating func resetGeneration() {
      self.index = 0
    }
  }
#endif
