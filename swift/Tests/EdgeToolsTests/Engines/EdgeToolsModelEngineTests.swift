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
      expectNoDifference(generation.prefillMetrics.tokens > 0, true)
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
      let constraint = EdgeToolsXGRGenerationConstraint.toolsWithGrammar(
        range: range,
        grammar: { toolsGrammar, _ in
          observation.recordTransform()
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
      let constraint = EdgeToolsXGRGenerationConstraint.toolsOrGrammar(
        userGrammar,
        range: range,
        transform: { toolsGrammar, _ in
          observation.recordTransform()
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

    var toolCallRange: GrammarToolCallRange? {
      self.range.withLock { $0 }
    }

    var didTransform: Bool {
      self.transformed.withLock { $0 }
    }

    func record(range: GrammarToolCallRange) {
      self.range.withLock { $0 = range }
    }

    func recordTransform() {
      self.transformed.withLock { $0 = true }
    }
  }

  private struct TestModel: EdgeToolsModel, Sendable {
    struct Parameters: EdgeToolsConstrainedGenerateParameters {
      static var `default`: Self { Parameters(tokenIds: []) }

      var tokenIds: [EdgeToolsToken.ID]
      var preparationDelay = Duration.zero
      var constraint = EdgeToolsXGRGenerationConstraint.unconstrained
      var maxTokens: Int? = 32
    }

    typealias Prompt = NeedlePrompt
    typealias Input = [EdgeToolsToken.ID]
    typealias GenerateParameters = Parameters
    typealias ToolCallParser = NeedleToolCallParser

    var assets: TestAssets?
    var constraintObservation: ConstraintObservation?
    var index = 0

    var vocabularySize: Int { 8192 }

    func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      self.constraintObservation?.record(range: range)
      return .universal
    }

    func input(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> EdgeToolsModelInput<[EdgeToolsToken.ID]> {
      let tokenIds = tokenizer.encode(text: prompt.user)
      return EdgeToolsModelInput(value: tokenIds, tokenIds: tokenIds)
    }

    nonisolated(nonsending) mutating func prepare(
      input: [EdgeToolsToken.ID],
      parameters: Parameters
    ) async throws -> EdgeToolsModelPreparation {
      self.assets?.begin()
      defer { self.assets?.end() }
      try await Task.sleep(for: parameters.preparationDelay)
      self.index = 0
      return EdgeToolsModelPreparation(
        metrics: EdgeToolsPrefillMetrics(tokens: input.count, duration: .zero)
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
