#if MLX && canImport(MLX) && !os(WASI)
  import CustomDump
  import EdgeTools
  import MLX
  import MLXLMCommon
  import MLXNN
  import Observation
  import Testing

  @Suite(.serialized)
  struct `MLXContext tests` {
    @Test
    func `Reasoning Effort And Tools Belong To The Context`() throws {
      let session = EdgeToolsSession(engine: try contextTestEngine())
      let context = session.context(
        systemPrompt: "System",
        reasoningEffort: .high
      ) {
        EchoTool()
      }

      expectNoDifference(context.transcript.messages, [.system("System")])
      expectNoDifference(context.reasoningEffort, .high)
      expectNoDifference(context.tools.map(\.name), ["echo"])

      let fork = context.fork()
      context.reasoningEffort = .none

      expectNoDifference(context.reasoningEffort, .none)
      expectNoDifference(fork.reasoningEffort, .high)
    }

    @Test
    func `Transcript Is Observable Through The Wrapper`() throws {
      let context = try contextTestEngine().context()
      let didChange = LockBox(false)

      withObservationTracking {
        _ = context.transcript
      } onChange: {
        didChange.withLock { $0 = true }
      }
      context.transcript.messages.append(.system("System"))

      expectNoDifference(didChange.withLock { $0 }, true)
    }

    @Test
    func `In Flight Transcript Mutation Does Not Change The Generation Snapshot`() async throws {
      await ContextTestProfile.gate.pauseNextCapture()
      let engine = try contextTestEngine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(
        MLXContextParameters(
          transcript: EdgeToolsTranscript(messages: [.system("System")])
        )
      )

      let generation = Task {
        try await session.generate(
          prompt: .user("Question"),
          context: context,
          parameters: MLXGenerateParameters(maxTokens: 1)
        )
      }
      let capturedTranscript = await ContextTestProfile.gate.waitForCapture()
      _ = try await session.tokenize(prompt: .user("Tokenize"), context: context)
      expectNoDifference(context.isResponding, true)
      context.transcript.messages.append(.system("Injected while generating"))
      await ContextTestProfile.gate.resume()
      _ = try await generation.value

      expectNoDifference(
        capturedTranscript.messages,
        [.system("System"), .user("Question")]
      )
      expectNoDifference(
        context.transcript.messages,
        [
          .system("System"),
          .user("Question"),
          .system("Injected while generating"),
          .assistant([])
        ]
      )
      expectNoDifference(context.isResponding, false)
    }

    @Test
    func `Fork Copies The Cache Lazily`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let copyCounter = CacheCopyCounter()
      let engine = try MLXEngine<ContextTestProfile>(
        languageModel: ForkTestLanguageModel(
          eosTokenId: eosTokenId,
          copyCounter: copyCounter
        ),
        tokenizer: tokenizer,
        vocabularySize: TestTokenizer.vocabularySize
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(
        MLXContextParameters(
          transcript: EdgeToolsTranscript(messages: [.system("System")])
        )
      )
      _ = try await engine.prefill(
        promptPrefix: EdgeToolsTranscript.Prompt(messages: []),
        context: context
      )
      expectNoDifference(copyCounter.count, 0)

      let idleFork = context.fork()
      expectNoDifference(copyCounter.count, 0)
      _ = try await session.generate(
        prompt: .user("Idle fork"),
        context: idleFork,
        parameters: MLXGenerateParameters(maxTokens: 1)
      )
      expectNoDifference(copyCounter.count, 1)

      await ContextTestProfile.gate.pauseNextCapture()
      let generation = Task {
        try await session.generate(
          prompt: .user("Question"),
          context: context,
          parameters: MLXGenerateParameters(maxTokens: 1)
        )
      }
      _ = await ContextTestProfile.gate.waitForCapture()
      expectNoDifference(context.isResponding, true)

      let fork = context.fork()

      expectNoDifference(copyCounter.count, 1)
      expectNoDifference(fork.isResponding, false)
      await ContextTestProfile.gate.resume()
      _ = try await generation.value
      expectNoDifference(copyCounter.count, 2)

      _ = try await session.generate(
        prompt: .user("Responding fork"),
        context: fork,
        parameters: MLXGenerateParameters(maxTokens: 1)
      )
      expectNoDifference(copyCounter.count, 3)
    }

    @Test
    func `Context Cannot Be Used With Another Engine`() async throws {
      let firstEngine = try contextTestEngine()
      let secondEngine = try contextTestEngine()
      let context = firstEngine.context()

      let error = await #expect(throws: EdgeToolsError.self) {
        _ = try await secondEngine.prefill(
          promptPrefix: EdgeToolsTranscript.Prompt(messages: []),
          context: context
        )
      }

      expectNoDifference(error?.code, .incompatibleContext)
    }
  }

  private struct ContextTestProfile: MLXLLMModelProfile {
    typealias Prompt = EdgeToolsTranscript
    typealias GenerationParser = TestGenerationParser
    typealias GrammarEngine = XGrammarEngine

    static let gate = TranscriptCaptureGate()

    static func grammar(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      parameters: MLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      .universal
    }

    static func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      await self.gate.capture(prompt)
      return LMInput(tokens: MLXArray(Array(0..<prompt.messages.count)))
    }

    static func prefillInput(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      try await self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        tokenizer: tokenizer,
        processor: processor
      )
    }
  }

  private actor TranscriptCaptureGate {
    private var shouldCapture = false
    private var capturedTranscript: EdgeToolsTranscript?
    private var captureWaiter: CheckedContinuation<EdgeToolsTranscript, Never>?
    private var resumeWaiter: CheckedContinuation<Void, Never>?

    func capture(_ transcript: EdgeToolsTranscript) async {
      guard self.shouldCapture else {
        return
      }
      self.shouldCapture = false
      self.capturedTranscript = transcript
      self.captureWaiter?.resume(returning: transcript)
      self.captureWaiter = nil
      await withCheckedContinuation { self.resumeWaiter = $0 }
    }

    func pauseNextCapture() {
      self.shouldCapture = true
      self.capturedTranscript = nil
    }

    func waitForCapture() async -> EdgeToolsTranscript {
      if let capturedTranscript = self.capturedTranscript {
        return capturedTranscript
      }
      return await withCheckedContinuation { self.captureWaiter = $0 }
    }

    func resume() {
      self.resumeWaiter?.resume()
      self.resumeWaiter = nil
    }
  }

  private final class CacheCopyCounter: Sendable {
    private let value = LockBox(0)

    var count: Int {
      self.value.withLock { $0 }
    }

    func increment() {
      self.value.withLock { $0 += 1 }
    }
  }

  private final class CountingKVCache: KVCache {
    private let copyCounter: CacheCopyCounter
    private let base: KVCacheSimple

    var offset: Int { self.base.offset }
    var maxSize: Int? { self.base.maxSize }
    var state: [MLXArray] {
      get { self.base.state }
      set { self.base.state = newValue }
    }
    var metaState: [String] {
      get { self.base.metaState }
      set { self.base.metaState = newValue }
    }
    var isTrimmable: Bool { self.base.isTrimmable }

    init(copyCounter: CacheCopyCounter, base: KVCacheSimple = KVCacheSimple()) {
      self.copyCounter = copyCounter
      self.base = base
    }

    func innerState() -> [MLXArray] {
      self.base.innerState()
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
      self.base.update(keys: keys, values: values)
    }

    func makeMask(
      n: Int,
      windowSize: Int?,
      returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
      self.base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    func trim(_ n: Int) -> Int {
      self.base.trim(n)
    }

    func copy() -> any KVCache {
      self.copyCounter.increment()
      return CountingKVCache(
        copyCounter: self.copyCounter,
        base: self.base.copy() as! KVCacheSimple
      )
    }
  }

  private final class ForkTestLanguageModel: Module, LanguageModel {
    let eosTokenId: EdgeToolsToken.ID
    let copyCounter: CacheCopyCounter
    var vocabularySize: Int { TestTokenizer.vocabularySize }

    init(eosTokenId: EdgeToolsToken.ID, copyCounter: CacheCopyCounter) {
      self.eosTokenId = eosTokenId
      self.copyCounter = copyCounter
      super.init()
    }

    func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
      let tokenCount = inputs.size
      let values = MLXArray.zeros([1, 1, tokenCount, 1])
      for cache in cache ?? [] {
        _ = cache.update(keys: values, values: values)
      }
      var logits = [Float](
        repeating: -100,
        count: tokenCount * self.vocabularySize
      )
      for index in 0..<tokenCount {
        logits[index * self.vocabularySize + self.eosTokenId] = 100
      }
      return MLXArray(logits, [1, tokenCount, self.vocabularySize])
    }

    func newCache(parameters: GenerateParameters?) -> [any KVCache] {
      [CountingKVCache(copyCounter: self.copyCounter)]
    }
  }

  private final class ContextTestLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
  {
    let eosTokenId: EdgeToolsToken.ID
    var vocabularySize: Int { TestTokenizer.vocabularySize }
    var kvHeads: [Int] { [1] }

    init(eosTokenId: EdgeToolsToken.ID) {
      self.eosTokenId = eosTokenId
      super.init()
    }

    func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
      var logits = [Float](repeating: -100, count: self.vocabularySize)
      logits[self.eosTokenId] = 100
      return MLXArray(logits, [1, 1, self.vocabularySize])
    }
  }

  private func contextTestEngine() throws -> MLXEngine<ContextTestProfile> {
    let tokenizer = try testTokenizer()
    let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
    return try MLXEngine(
      languageModel: ContextTestLanguageModel(eosTokenId: eosTokenId),
      tokenizer: tokenizer,
      vocabularySize: TestTokenizer.vocabularySize
    )
  }
#endif
