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
          decode: { _, batch in
            batches.withLock { $0.append(batch) }
            return 0
          },
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
    func `Forked Context Shares KV Cells Copy On Write`() async throws {
      let logits = ScriptedLogits(script: [3, 2], vocabularySize: 6)
      let batches = LockBox([LlamaDecodeBatch]())
      let copies = LockBox([[Int]]())
      let engine = try LlamaEngine<ScriptLlamaProfile>(
        api: mockLlamaApi(
          decode: { _, batch in
            batches.withLock { $0.append(batch) }
            return 0
          },
          lastLogits: { _ in logits.next() },
          onMemoryCopy: { source, destination in
            copies.withLock { $0.append([Int(source), Int(destination)]) }
          }
        ),
        modelPath: "mock"
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(
        transcript: EdgeToolsTranscript(messages: [.user("hello world")])
      )
      _ = try await engine.prefill(context: context)

      let fork = context.fork()
      _ = try await session.generate(
        prompt: .user(""),
        context: fork,
        parameters: DefaultLlamaGenerateParameters(sampling: .greedy)
      )

      expectNoDifference(copies.withLock { $0 }, [[0, 1]])
      let forkBatches = batches.withLock { Array($0.dropFirst()) }
      expectNoDifference(forkBatches.allSatisfy { $0.sequenceId == 1 }, true)
      expectNoDifference(forkBatches.first.map { [$0.startPosition] + $0.tokens }, [2, 5])
    }

    @Test
    func `Fork Beyond Sequence Capacity Falls Back To A Fresh Family`() async throws {
      let logits = ScriptedLogits(script: [3, 2], vocabularySize: 6)
      let batches = LockBox([LlamaDecodeBatch]())
      let contextCount = LockBox(0)
      let engine = try LlamaEngine<ScriptLlamaProfile>(
        api: mockLlamaApi(
          decode: { _, batch in
            batches.withLock { $0.append(batch) }
            return 0
          },
          lastLogits: { _ in logits.next() },
          onCreateContext: { contextCount.withLock { $0 += 1 } }
        ),
        modelPath: "mock",
        contextParameters: LlamaContextParameters(maximumSequenceCount: 1)
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(
        transcript: EdgeToolsTranscript(messages: [.user("hello world")])
      )
      _ = try await engine.prefill(context: context)

      let fork = context.fork()
      _ = try await session.generate(
        prompt: .user(""),
        context: fork,
        parameters: DefaultLlamaGenerateParameters(sampling: .greedy)
      )

      expectNoDifference(contextCount.withLock { $0 }, 2)
      let forkBatches = batches.withLock { Array($0.dropFirst()) }
      expectNoDifference(forkBatches.allSatisfy { $0.sequenceId == 0 }, true)
      expectNoDifference(forkBatches.first.map { $0.startPosition }, 0)
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
