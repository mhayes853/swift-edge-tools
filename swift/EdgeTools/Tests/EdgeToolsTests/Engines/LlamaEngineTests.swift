#if LlamaCore && XGrammar
  import CustomDump
  import EdgeTools
  import EdgeToolsXGrammar
  import Testing

  @Suite
  struct `LlamaEngine tests` {
    @Test
    func `Generates A Scripted Response Through The Session`() async throws {
      let logits = ScriptedLogits(script: [3, 4, 2], vocabularySize: 6)
      let engine = try LlamaEngine<ScriptLlamaProfile>(
        api: mockLlamaApi(lastLogits: { _ in logits.next() }),
        modelPath: "mock"
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(transcript: EdgeToolsTranscript())

      let generation = try await session.generate(
        prompt: .user("hello world"),
        context: context,
        parameters: DefaultLlamaGenerateParameters(sampling: .greedy)
      )

      expectNoDifference(generation.text, " hello world")
      expectNoDifference(generation.engineGeneration.tokens.map(\.id), [3, 4, 2])
      expectNoDifference(generation.engineGeneration.prefillMetrics.tokens, 3)
      expectNoDifference(
        generation.engineGeneration.metadata.generationConfidence != nil,
        true
      )
      expectNoDifference(context.isResponding, false)
      expectNoDifference(context.transcript.messages.count, 2)
    }

    @Test
    func `Prefill Reuses The KV Cache For The Following Generation`() async throws {
      let logits = ScriptedLogits(script: [3, 2], vocabularySize: 6)
      let batches = LockBox([LlamaDecodeBatch]())
      let engine = try LlamaEngine<ScriptLlamaProfile>(
        api: mockLlamaApi(
          decode: { _, batch in batches.withLock { $0.append(batch) } },
          lastLogits: { _ in logits.next() }
        ),
        modelPath: "mock"
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(
        transcript: EdgeToolsTranscript(messages: [.user("hello world")])
      )

      let prefill = try await engine.prefill(context: context)
      expectNoDifference(prefill.metrics.tokens, 2)
      expectNoDifference(
        batches.withLock { $0.map { [$0.startPosition] + $0.tokens } },
        [[0, 3, 4]]
      )

      _ = try await session.generate(
        prompt: .user(""),
        context: context,
        parameters: DefaultLlamaGenerateParameters(sampling: .greedy)
      )
      let generationBatches = batches.withLock { Array($0.dropFirst()) }
      expectNoDifference(generationBatches.first.map { [$0.startPosition] + $0.tokens }, [2, 5])
      expectNoDifference(generationBatches.allSatisfy { $0.startPosition >= 2 }, true)
    }

    @Test
    func `Grammar Bitmask Excludes Tokens From Sampling`() async throws {
      let logits = ScriptedLogits(script: [3, 2], vocabularySize: 6)
      let engine = try LlamaEngine<ScriptLlamaProfile>(
        api: mockLlamaApi(lastLogits: { _ in logits.next() }),
        modelPath: "mock"
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(transcript: EdgeToolsTranscript())

      let generation = try await session.generate(
        prompt: .user("hello"),
        context: context,
        parameters: DefaultLlamaGenerateParameters(sampling: .greedy, maxTokens: 4)
      )

      expectNoDifference(generation.engineGeneration.tokens.map(\.id), [3, 2])
      expectNoDifference(generation.engineGeneration.wasStopped, false)
    }
  }

  // MARK: - ScriptLlamaProfile

  private struct ScriptLlamaProfile: LlamaModelProfile {
    typealias Prompt = EdgeToolsTranscript
    typealias GenerateParameters = DefaultLlamaGenerateParameters
    typealias GenerationParser = TestGenerationParser
    typealias GrammarEngine = XGrammarEngine

    static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultLlamaGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      .universal
    }
  }

  // MARK: - ScriptedLogits

  private final class ScriptedLogits: @unchecked Sendable {
    private let storage: UnsafeMutableBufferPointer<Float>
    private let script: [Int]
    private let index = LockBox(0)

    init(script: [Int], vocabularySize: Int) {
      self.storage = UnsafeMutableBufferPointer.allocate(capacity: vocabularySize)
      self.script = script
    }

    deinit { self.storage.deallocate() }

    func next() -> UnsafeMutablePointer<Float> {
      let step = self.index.withLock { index in
        defer { index += 1 }
        return index
      }
      for lane in self.storage.indices {
        self.storage[lane] = -10
      }
      self.storage[self.script[min(step, self.script.count - 1)]] = 10
      return self.storage.baseAddress!
    }
  }
#endif
