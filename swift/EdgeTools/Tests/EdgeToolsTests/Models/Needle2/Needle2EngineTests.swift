#if Needle2 && (os(macOS) || os(Linux) || os(Windows) || os(Android))
  import CustomDump
  import EdgeTools
  import Observation
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .extendedNeedleInference())
  struct `Needle2Engine tests` {
    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Generates Tool Call Through Session`() async throws {
      let session = EdgeToolsSession(engine: Needle2Engine()) {
        SendEmailTool()
      }

      let generation = try await session.generate(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        context: nil
      )

      expectNoDifference(generation.engineGeneration.wasStopped, false)
      expectNoDifference(generation.engineGeneration.tokens, [])
      expectNoDifference(generation.engineGeneration.toolCalls.count, 1)
      expectNoDifference(generation.engineGeneration.toolCalls.first?.name, "send_email")
      expectNoDifference(generation.toolCalls.count, 1)
      withKnownIssue {
        assertSnapshot(of: generation.engineGeneration.parts, as: .dump, record: .all)
      }
      expectNoDifference(generation.engineGeneration.metadata.generationConfidence != nil, true)
      expectNoDifference(
        generation.engineGeneration.metadata.needle2PrefillTokensPerSecond.map { $0 > 0 },
        true
      )
      expectNoDifference(
        generation.engineGeneration.metadata.needle2DecodeTokensPerSecond.map { $0 > 0 },
        true
      )
      expectNoDifference(
        generation.engineGeneration.metadata.needle2PeakRAMMegabytes.map { $0 > 0 },
        true
      )
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Stopped Generation Completes With Its Response`() async throws {
      let engine = Needle2Engine()
      let context = engine.context()
      let emittedParts = Lock([EdgeToolsGenerationPart]())
      let didObserveResponse = Lock(false)
      withObservationTracking {
        _ = context.isResponding
      } onChange: {
        didObserveResponse.withLock { $0 = true }
      }
      let task = try engine.generate(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        tools: [SendEmailTool().definition],
        parameters: .default,
        context: context,
        channel: EdgeToolsGenerationChannel(
          onPart: { part in emittedParts.withLock { $0.append(part) } }
        )
      )

      while !context.isResponding {
        await Task.yield()
      }
      expectNoDifference(context.isResponding, true)
      expectNoDifference(didObserveResponse.withLock { $0 }, true)
      task.stop()
      let generation = try await task.value

      expectNoDifference(context.isResponding, false)
      expectNoDifference(generation.wasStopped, true)
      expectNoDifference(generation.toolCalls.count, 1)
      expectNoDifference(emittedParts.withLock { $0 }, [])
      expectNoDifference(generation.metadata.needle2PeakRAMMegabytes != nil, true)
    }
  }
#endif
