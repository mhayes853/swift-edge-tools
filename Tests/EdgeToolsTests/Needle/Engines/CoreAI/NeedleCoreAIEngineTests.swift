#if swift(>=6.4) && CoreAI && Sentencepiece && canImport(CoreAI)
  import CoreAI
  import CustomDump
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .experimental())
  struct `NeedleCoreAIEngine tests` {
    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokens.withLock { $0.append(token) } }
        )
      )
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
    func `Concurrent Generations`() async throws {
      let engine = try await makeNeedleCoreAIEngine()

      let t1 = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let t2 = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )

      let (g1, g2) = try await (t1.value, t2.value)
      withKnownIssue {
        assertSnapshot(of: g1.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: g2.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Sequential Generations`() async throws {
      let engine = try await makeNeedleCoreAIEngine()

      let t1 = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let g1 = try await t1.value

      let t2 = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let g2 = try await t2.value

      withKnownIssue {
        assertSnapshot(of: g1.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: g2.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Streamed Response Matches Final Response`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokens.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value

      let streamedResponse = tokens.withLock { $0.map(\.stringValue).joined() }
      let finalResponse = generation.tokens.map(\.stringValue).joined()
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test(.disabled("TODO - Investigate Metal Crash Issue"))
    @available(anyAppleOS 27.0, *)
    func `Generate With Compute Stream`() async throws {
      let engine = try await makeNeedleCoreAIEngine()

      let params = NeedleCoreAIEngine.GenerateParameters(computeStream: ComputeStream())
      let task = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: params,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await task.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let tokens = Lock([EdgeToolsToken]())
      let generationTaskBox = Lock<NeedleCoreAIEngine.GenerationTask?>(nil)
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in
            tokens.withLock { $0.append(token) }
            generationTaskBox.withLock { $0?.stop() }
          }
        )
      )
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
          parameters: .default,
          channel: EdgeToolsGenerationChannel()
        )
        _ = try await generationTask.value
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Basics With W8 Quantized Export`() async throws {
      let engine = try await makeNeedleCoreAIEngine(quantizerPreset: "w8")
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokens.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test(.disabled("TODO - Apple seems to have broken something here in the latest beta."))
    @available(anyAppleOS 27.0, *)
    func `Generate Basics With AOT Compiled Export`() async throws {
      let compilePlatform: String = {
        #if os(macOS)
          "macOS"
        #elseif os(iOS)
          "iOS"
        #elseif os(tvOS)
          "tvOS"
        #elseif os(visionOS)
          "visionOS"
        #else
          "watchOS"
        #endif
      }()
      let engine = try await makeNeedleCoreAIEngine(compilePlatforms: [compilePlatform])
      let task = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await task.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Through EdgeToolsSession`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let session = EdgeToolsSession(engine: engine)
      let generation = try await session.generate(
        tools: [SendEmailTool()],
        with: EdgeToolsPrompt.sendAdventureEmail.user
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
        parameters: NeedleCoreAIEngine.GenerateParameters(processor: processor),
        channel: EdgeToolsGenerationChannel()
      )
      _ = try await generationTask.value

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let engine = try await makeNeedleCoreAIEngine()
      let prompt = EdgeToolsPrompt(
        system: "",
        user: String(repeating: "token ", count: 2_000),
        tools: [.sendEmail]
      )

      let error = await #expect(throws: NeedleCoreAIEngineError.self) {
        let generationTask = try engine.generate(
          prompt: prompt,
          parameters: .default,
          channel: EdgeToolsGenerationChannel()
        )
        _ = try await generationTask.value
      }
      expectNoDifference(error?.message.contains("context length"), true)
    }
  }

  @available(anyAppleOS 27.0, *)
  final class CountingLogitsProcessor: EdgeToolsLogitsProcessor, @unchecked Sendable {
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

    func didSample(token: EdgeToolsToken) {
      self.didSampleCalls += 1
    }
  }
#endif
