#if XGrammar && Foundation
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `EdgeToolsGenerationLoop tests` {
    @Test
    func `Generates Through The Shared Generation Loop`() async throws {
      let tokenizer = try testTokenizer()
      let responseTokenIds = encodedGrammarText("hello", tokenizer: tokenizer)
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let engine = try TestEngine(tokenizer: tokenizer)
      let task = try engine.generate(
        prompt: TestPrompt(system: "", user: "Prompt"),
        parameters: TestEngine.Parameters(tokenIds: responseTokenIds + [eosTokenId]),
        channel: EdgeToolsGenerationChannel()
      )

      let generation = try await task.value

      expectNoDifference(generation.wasStopped, false)
      expectNoDifference(generation.tokens.map(\.id), responseTokenIds + [eosTokenId])
      expectNoDifference(generation.response, tokenizer.decode(tokens: responseTokenIds))
      expectNoDifference(generation.text, tokenizer.decode(tokens: responseTokenIds))
      expectNoDifference(generation.prefillMetrics.tokens > 0, true)
      expectNoDifference(generation.metadata.generationConfidence, nil)
      expectNoDifference(generation.metadata.perTokenConfidences, nil)
    }

    @Test
    func `Stops At Any Extra Model Stop Token`() async throws {
      let tokenizer = try testTokenizer()
      let responseTokenIds = encodedGrammarText("hello", tokenizer: tokenizer)
      let stopTokenId = try #require(tokenizer.unknownTokenId)
      let alternateStopTokenId = try #require(tokenizer.bosTokenId)
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let engine = try TestEngine(
        tokenizer: tokenizer,
        extraStopTokenIds: [alternateStopTokenId, stopTokenId]
      )
      let task = try engine.generate(
        prompt: TestPrompt(system: "", user: "Prompt"),
        parameters: TestEngine.Parameters(
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
      let engine = try TestEngine(tokenizer: tokenizer, constraintObservation: observation)
      let range = GrammarToolCallRange.exact(1)
      let constraint = XGRGenerationConstraint.toolsWithGrammar(
        range: range,
        grammar: { toolsGrammar, tokenizerInfo in
          observation.recordTransform(tokenizerInfo: tokenizerInfo)
          return try toolsGrammar.copy()
        }
      )
      let task = try engine.generate(
        prompt: TestPrompt(system: "", user: "Prompt"),
        parameters: TestEngine.Parameters(
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
    func `Runs Different Contexts Concurrently`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let assets = TestAssets()
      let engine = try TestEngine(tokenizer: tokenizer, assets: assets)
      let parameters = TestEngine.Parameters(
        tokenIds: [eosTokenId],
        preparationDelay: .milliseconds(50)
      )
      let first = try engine.generate(
        prompt: TestPrompt(system: "", user: "First"),
        parameters: parameters,
        context: engine.context(),
        channel: EdgeToolsGenerationChannel()
      )
      let second = try engine.generate(
        prompt: TestPrompt(system: "", user: "Second"),
        parameters: parameters,
        context: engine.context(),
        channel: EdgeToolsGenerationChannel()
      )

      async let firstGeneration = first.value
      async let secondGeneration = second.value
      _ = try await (firstGeneration, secondGeneration)

      expectNoDifference(assets.maximumActiveCount, 2)
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

  private struct TestGenerationState: Sendable {
    var index = 0
  }

  private final class TestContext: Identifiable, Sendable {
    private actor Runtime {
      var state: TestGenerationState? = TestGenerationState()

      func takeState() throws -> sending TestGenerationState {
        guard let state = self.state.take() else {
          throw CancellationError()
        }
        return state
      }

      func restore(state: sending TestGenerationState) {
        self.state = state
      }
    }

    private let runtime = Runtime()

    var id: ObjectIdentifier {
      ObjectIdentifier(self)
    }

    func takeState() async throws -> sending TestGenerationState {
      try await self.runtime.takeState()
    }

    func restore(state: sending TestGenerationState) async {
      await self.runtime.restore(state: state)
    }
  }

  private final class TestEngine: EdgeToolsEngine {
    typealias Context = TestContext
    typealias ContextParameters = Void
    typealias Prompt = TestPrompt
    typealias GenerateParameters = Parameters
    typealias ModelGenerationState = TestGenerationState
    typealias GenerationParser = TestGenerationParser
    typealias GrammarEngine = XGrammarEngine

    struct Parameters: EdgeToolsConstrainedGenerateParameters {
      static var `default`: Self { Parameters(tokenIds: []) }

      var tokenIds: [EdgeToolsToken.ID]
      var preparationDelay = Duration.zero
      var constraint = XGRGenerationConstraint.unconstrained
      var maxTokens: Int? = 32
    }

    let assets: TestAssets?
    let constraintObservation: ConstraintObservation?
    let generationLoop: EdgeToolsGenerationLoop
    let tokenizer: any EdgeToolsTokenizer
    let grammarEngine: XGrammarEngine

    init(
      tokenizer: any EdgeToolsTokenizer,
      assets: TestAssets? = nil,
      constraintObservation: ConstraintObservation? = nil,
      extraStopTokenIds: Set<EdgeToolsToken.ID> = []
    ) throws {
      guard let tokenizer = tokenizer as? any XGRTokenizer else {
        throw XGRError(
          code: .invalidTokenizerInfo,
          message: "The test engine requires an XGrammar tokenizer."
        )
      }
      self.grammarEngine = try XGrammarEngine(
        tokenizerInfo: tokenizer.tokenizerInfo(
          modelVocabularySize: TestTokenizer.vocabularySize,
          extraStopTokenIds: extraStopTokenIds
        )
      )
      self.tokenizer = tokenizer
      self.assets = assets
      self.constraintObservation = constraintObservation
      self.generationLoop = EdgeToolsGenerationLoop(
        tokenizer: tokenizer,
        extraStopTokenIds: extraStopTokenIds
      )
    }

    func context(_ parameters: Void) -> TestContext {
      TestContext()
    }

    func grammar(
      prompt: TestPrompt,
      tools: [EdgeToolDefinition],
      parameters: Parameters,
      state: TestGenerationState
    ) throws -> XGRGrammar {
      let constraint = parameters.constraint
      let grammar = constraint.toolCallRange.map {
        self.constraintObservation?.record(range: $0)
        return XGRGrammar.universal
      }
      return try constraint.grammar(toolCallGrammar: grammar, context: self.grammarEngine)
    }

    func tokenize(
      prompt: TestPrompt,
      tools: [EdgeToolDefinition],
      context: TestContext
    ) async throws -> [EdgeToolsToken] {
      let tokenIds = self.tokenizer.encode(text: prompt.user)
      return zip(tokenIds, self.tokenizer.convertIdsToTokens(tokenIds))
        .compactMap {
          tokenId,
          token in token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
        }
    }

    func generationState(
      prompt: TestPrompt,
      context: TestContext
    ) async throws -> TestGenerationState {
      try await context.takeState()
    }

    func generate(
      prompt: TestPrompt,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Parameters,
      context: TestContext,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      AnyGenerationTask { stopper in
        var state = try await self.generationState(prompt: prompt, context: context)
        var preparedPrompt = prompt
        let result: Result<EdgeToolsEngineGeneration, any Error>
        do {
          result = .success(
            try await self.generationLoop.run(
              state: &state,
              stopper: stopper,
              channel: channel,
              grammarEngine: self.grammarEngine,
              maximumTokenCount: parameters.maxTokens,
              grammar: { state in
                try self.grammar(
                  prompt: preparedPrompt,
                  tools: tools,
                  parameters: parameters,
                  state: state
                )
              },
              prepare: { parser, state in
                return try await self.prepare(
                  prompt: &preparedPrompt,
                  tools: tools,
                  parameters: parameters,
                  parser: &parser,
                  state: &state
                )
              },
              decode: { bitmask, state in
                try await self.decode(
                  bitmask: bitmask,
                  parameters: parameters,
                  state: &state
                )
              }
            )
          )
        } catch {
          result = .failure(error)
        }
        return try await self.finalize(state: state, result: result, context: context).get()
      }
    }

    func prepare(
      prompt: inout TestPrompt,
      tools: [EdgeToolDefinition],
      parameters: Parameters,
      parser: inout TestGenerationParser,
      state: inout TestGenerationState
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
      let tokenIds = self.tokenizer.encode(text: prompt.user)
      self.assets?.begin()
      defer { self.assets?.end() }
      try await Task.sleep(for: parameters.preparationDelay)
      state.index = 0
      return EdgeToolsGenerationLoop.Preparation(
        metrics: EdgeToolsPrefillMetrics(tokens: tokenIds.count, duration: .zero)
      )
    }

    func decode(
      bitmask: GrammarBitmask,
      parameters: Parameters,
      state: inout TestGenerationState
    ) async throws -> EdgeToolsToken.ID {
      let tokenId = parameters.tokenIds[state.index]
      state.index += 1
      return tokenId
    }

    func finalize(
      state: consuming TestGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: TestContext
    ) async -> Result<EdgeToolsEngineGeneration, any Error> {
      await context.restore(state: state)
      return result
    }
  }
#endif
