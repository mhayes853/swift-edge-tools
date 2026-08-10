#if CoreML && canImport(CoreML) && !os(WASI)
  import CoreML
  import CustomDump
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .experimental(), .extendedNeedleInference())
  struct `NeedleCoreMLModelEngine tests` {
    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine(computeUnits: .cpuOnly)
      let generation = try await generateNeedle(using: engine)

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
      }
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Concurrent Generations`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      let (first, second) = try await generateNeedleConcurrently(using: engine)
      withKnownIssue {
        assertSnapshot(of: first.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: second.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Sequential Generations`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      let (first, second) = try await generateNeedleSequentially(using: engine)
      withKnownIssue {
        assertSnapshot(of: first.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: second.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Streamed Response Matches Final Response`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      try await expectNeedleStreamedResponseMatchesFinalResponse(using: engine)
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      try await expectNeedleGenerationStops(using: engine)
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      await expectNeedleGenerationCancellation(using: engine)
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics With W8 Quantized Export`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine(quantizerPreset: "w8")
      let generation = try await generateNeedle(using: engine)

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
      let engine = try await makeNeedleCoreMLModelEngine(
        quantizerPreset: "w8",
        palettizerBits: 4
      )
      let generation = try await generateNeedle(using: engine)

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
      let engine = try await makeNeedleCoreMLModelEngine(compilePlatforms: [compilePlatform])
      let generation = try await generateNeedle(using: engine)

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Through EdgeToolsSession`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      let generation = try await generateNeedleThroughSession(using: engine)

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
      let engine = try await makeNeedleCoreMLModelEngine()
      let processor = CountingCoreMLLogitsProcessor()
      _ = try await generateNeedle(
        using: engine,
        parameters: NeedleCoreMLModel.GenerateParameters(processor: processor.value)
      )

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      await expectNeedleContextLengthExceeded(using: engine)
    }
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  final class CountingCoreMLLogitsProcessor: Sendable {
    private struct Counts: Hashable, Sendable {
      var prompt = 0
      var process = 0
      var didSample = 0
    }

    private let counts = Lock(Counts())

    var promptCalls: Int { self.counts.withLock { $0.prompt } }
    var processCalls: Int { self.counts.withLock { $0.process } }
    var didSampleCalls: Int { self.counts.withLock { $0.didSample } }

    var value: EdgeToolsLogitsProcessor<MLTensor, MLTensor> {
      EdgeToolsLogitsProcessor(
        prompt: { self.prompt($0) },
        process: { try await self.process(logits: &$0) },
        didSample: { self.didSample(tokenId: $0) }
      )
    }

    func prompt(_ prompt: MLTensor) {
      self.counts.withLock { $0.prompt += 1 }
    }

    func process(logits: inout MLTensor) async throws {
      self.counts.withLock { $0.process += 1 }
    }

    func didSample(tokenId: EdgeToolsToken.ID) {
      self.counts.withLock { $0.didSample += 1 }
    }
  }
#endif
