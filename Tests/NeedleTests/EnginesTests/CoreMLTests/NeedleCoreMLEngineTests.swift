#if swift(>=6.4) && CoreML && Sentencepiece && canImport(CoreML)
  import CoreML
  import CustomDump
  import Needle
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .experimental())
  struct `NeedleCoreMLEngine tests` {
    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Sequential Generations`() async throws {
      let engine = try await makeNeedleCoreMLEngine()

      let t1 = try engine.generate(prompt: .sendAdventureEmail, parameters: .default) { _ in }
      let g1 = try await t1.value

      let t2 = try engine.generate(prompt: .sendAdventureEmail, parameters: .default) { _ in }
      let generation2 = try await t2.value

      withKnownIssue {
        assertSnapshot(of: g1.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: generation2.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Streamed Response Matches Final Response`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let tokens = Lock([NeedleToken]())
      let generationTaskBox = Lock<NeedleCoreMLEngine.GenerationTask?>(nil)
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
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

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics With W8 Quantized Export`() async throws {
      let engine = try await makeNeedleCoreMLEngine(quantizerPreset: "w8")
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Through NeedleSession`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Invokes Custom Logit Processor`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let processor = CountingCoreMLLogitsProcessor()
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: NeedleCoreMLEngine.GenerateParameters(processor: processor)
      ) { _ in }
      _ = try await generationTask.value

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let prompt = NeedlePrompt(
        system: "",
        user: String(repeating: "token ", count: 2_000),
        tools: [.sendEmail]
      )

      let error = await #expect(throws: NeedleCoreMLEngineError.self) {
        let generationTask = try engine.generate(prompt: prompt, parameters: .default) { _ in }
        _ = try await generationTask.value
      }
      expectNoDifference(error?.message.contains("context length"), true)
    }
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  final class CountingCoreMLLogitsProcessor: NeedleCoreMLEngine.LogitsProcessor, @unchecked Sendable {
    var promptCalls = 0
    var processCalls = 0
    var didSampleCalls = 0

    func prompt(_ prompt: MLTensor) {
      self.promptCalls += 1
    }

    func process(_ logits: MLTensor) async throws -> MLTensor {
      self.processCalls += 1
      return logits
    }

    func didSample(token: NeedleToken) {
      self.didSampleCalls += 1
    }
  }
#endif
