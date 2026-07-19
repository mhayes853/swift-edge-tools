#if CoreML && canImport(CoreML)
  import CoreML
  import CustomDump
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .experimental())
  struct `NeedleCoreMLEngine tests` {
    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
      }
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Concurrent Generations`() async throws {
      let engine = try await makeNeedleCoreMLEngine()

      let t1 = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let t2 = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Sequential Generations`() async throws {
      let engine = try await makeNeedleCoreMLEngine()

      let t1 = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let g1 = try await t1.value

      let t2 = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
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
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokens.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value

      let streamedResponse = tokens.withLock { $0.map(\.stringValue).joined() }
      let finalResponse = generation.response
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let tokens = Lock([EdgeToolsToken]())
      let generationTaskBox = Lock<NeedleCoreMLEngine.GenerationTask?>(nil)
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let task = Task {
        let generationTask = try engine.generate(
          prompt: .sendAdventureEmail,
          tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics With W8 Quantized Export`() async throws {
      let engine = try await makeNeedleCoreMLEngine(quantizerPreset: "w8")
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics With W8 Quantized And W4 Palettized Export`() async throws {
      let engine = try await makeNeedleCoreMLEngine(
        quantizerPreset: "w8",
        palettizerBits: 4
      )
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics With AOT Compiled Export`() async throws {
      let compilePlatform: String = {
        #if targetEnvironment(macCatalyst)
          "macCatalyst"
        #elseif os(macOS)
          "macOS"
        #elseif os(iOS)
          "iOS"
        #elseif os(watchOS)
          "watchOS"
        #elseif os(tvOS)
          "tvOS"
        #else
          "visionOS"
        #endif
      }()
      let engine = try await makeNeedleCoreMLEngine(compilePlatforms: [compilePlatform])
      let task = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
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
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Through EdgeToolsSession`() async throws {
      let engine = try await makeNeedleCoreMLEngine()
      let session = EdgeToolsSession(engine: engine)
      let generation = try await session.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailTools
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
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: NeedleCoreMLEngine.GenerateParameters(processor: processor),
        channel: EdgeToolsGenerationChannel()
      )
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
        user: String(repeating: "token ", count: 2_000)
      )

      let error = await #expect(throws: NeedleModelError.self) {
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

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  final class CountingCoreMLLogitsProcessor: EdgeToolsLogitsProcessor, @unchecked Sendable {
    var promptCalls = 0
    var processCalls = 0
    var didSampleCalls = 0

    func prompt(_ prompt: MLTensor) {
      self.promptCalls += 1
    }

    func process(logits: inout MLTensor) async throws -> MLTensor {
      self.processCalls += 1
      return logits
    }

    func didSample(token: EdgeToolsToken) {
      self.didSampleCalls += 1
    }
  }
#endif
