#if swift(>=6.4) && CoreAI && Sentencepiece && canImport(CoreAI)
  import CoreAI
  import CustomDump
  import Needle
  import SnapshotTesting
  import Testing

  struct `NeedleCoreAIEngine tests` {
    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let tokens = Lock([NeedleToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { token in tokens.withLock { $0.append(token) } }
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Streamed Response Matches Final Response`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let tokens = Lock([NeedleToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { token in tokens.withLock { $0.append(token) } }
      let generation = try await generationTask.value

      let streamedResponse = tokens.withLock { $0.map(\.stringValue).joined() }
      let finalResponse = generation.tokens.map(\.stringValue).joined()
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let tokens = Lock([NeedleToken]())
      let generationTaskBox = Lock<NeedleCoreAIEngine.GenerationTask?>(nil)
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { token in
        tokens.withLock { $0.append(token) }
        generationTaskBox.withLock { $0?.stop() }
      }
      generationTaskBox.withLock { $0 = generationTask }
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, true)
      let tokenCount = tokens.withLock { $0.count }
      expectNoDifference(tokenCount > 0, true)
      expectNoDifference(generation.decodeMetrics.tokens, tokenCount)
      expectNoDifference(generation.tokens.isEmpty, false)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let task = Task {
        let generationTask = try engine.generate(
          prompt: .sendAdventureEmail,
          parameters: .default
        ) { _ in }
        _ = try await generationTask.value
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test(.enabledIfXcode())
    @available(anyAppleOS 27.0, *)
    func `Generate Through NeedleSession`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let session = NeedleSession(engine: engine)
      let generation = try await session.generate(
        tools: [SendEmailTool()],
        with: NeedlePrompt.sendAdventureEmail.user
      )

      expectNoDifference(generation.engineGeneration.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.engineGeneration.metadata, as: .dump, record: .all)
        assertSnapshot(
          of: generation.engineGeneration.tokens.map(\.stringValue).joined(),
          as: .lines,
          record: .all
        )
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Invokes Custom Logit Processor`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let processor = CountingLogitsProcessor()
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: NeedleCoreAIEngine.GenerateParameters(processor: processor)
      ) { _ in }
      _ = try await generationTask.value

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let prompt = NeedlePrompt(
        system: "",
        user: String(repeating: "token ", count: 2_000),
        tools: [.sendEmail]
      )

      let error = await #expect(throws: NeedleCoreAIEngineError.self) {
        let generationTask = try engine.generate(prompt: prompt, parameters: .default) { _ in }
        _ = try await generationTask.value
      }
      expectNoDifference(error?.message.contains("context length"), true)
    }
  }

  @available(anyAppleOS 27.0, *)
  final class CountingLogitsProcessor: NeedleCoreAIEngine.LogitsProcessor, @unchecked Sendable {
    var promptCalls = 0
    var processCalls = 0
    var didSampleCalls = 0

    func prompt(_ prompt: NDArray) {
      self.promptCalls += 1
    }

    func process(logits: inout NDArray) -> NDArray {
      self.processCalls += 1
      return logits
    }

    func didSample(token: NeedleToken) {
      self.didSampleCalls += 1
    }
  }
#endif
