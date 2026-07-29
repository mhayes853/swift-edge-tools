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
        assets: TestAssets(),
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
    func `Resolves Tool Constraints Inside The Model Engine`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let observation = ConstraintObservation()
      let model = TestModel(constraintObservation: observation)
      let engine = try EdgeToolsModelEngine(
        model: model,
        assets: TestAssets(),
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
    func `Serializes Concurrent Generations`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let assets = TestAssets()
      let engine = try EdgeToolsModelEngine(
        model: TestModel(),
        assets: assets,
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
    struct Parameters: EdgeToolsModelEngineGenerateParameters {
      static var `default`: Self { Parameters(tokenIds: []) }

      var tokenIds: [EdgeToolsToken.ID]
      var preparationDelay = Duration.zero
      var constraint = EdgeToolsXGRGenerationConstraint.unconstrained
      var maxTokens: Int? = 32
    }

    struct State: Hashable, Sendable {
      var tokenIds: [EdgeToolsToken.ID]
      var index = 0
    }

    typealias Prompt = NeedlePrompt
    typealias Input = [EdgeToolsToken.ID]
    typealias Logits = [Float]
    typealias GenerationState = State
    typealias Assets = TestAssets
    typealias GenerateParameters = Parameters
    typealias ToolCallParser = NeedleToolCallParser

    var constraintObservation: ConstraintObservation?

    var vocabularySize: Int { 8192 }

    func grammar(
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
    ) throws -> [EdgeToolsToken.ID] {
      tokenizer.encode(text: prompt.user)
    }

    func tokenIds(in input: [EdgeToolsToken.ID]) -> [EdgeToolsToken.ID] {
      input
    }

    nonisolated(nonsending) func prepare(
      input: [EdgeToolsToken.ID],
      parameters: Parameters,
      assets: TestAssets
    ) async throws -> EdgeToolsModelPreparation<[Float], State> {
      assets.begin()
      defer { assets.end() }
      try await Task.sleep(for: parameters.preparationDelay)
      return EdgeToolsModelPreparation(
        logits: [],
        state: State(tokenIds: parameters.tokenIds),
        metrics: EdgeToolsPrefillMetrics(tokens: input.count, duration: .zero)
      )
    }

    nonisolated(nonsending) func decode(
      tokenId: EdgeToolsToken.ID,
      state: inout State,
      assets: TestAssets
    ) async throws -> [Float] {
      []
    }

    nonisolated(nonsending) func sample(
      logits: inout [Float],
      bitmask: GrammarBitmask,
      state: inout State
    ) async throws -> EdgeToolsModelSample {
      let tokenId = state.tokenIds[state.index]
      state.index += 1
      return EdgeToolsModelSample(tokenId: tokenId, confidence: 1)
    }

    func didAccept(token: EdgeToolsToken, state: inout State) {}
  }
#endif
