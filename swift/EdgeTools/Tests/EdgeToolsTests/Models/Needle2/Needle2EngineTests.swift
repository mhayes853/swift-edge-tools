#if Needle2 && (os(macOS) || os(Linux) || os(Windows) || os(Android))
  import CustomDump
  import EdgeTools
  import Observation
  import SnapshotTesting
  import Testing

  @Suite(.serialized)
  struct `Needle2Engine tests` {
    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Generates Tool Call Through Session`() async throws {
      let engine = Needle2Engine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context { SendEmailTool() }
      defer { try? engine.reset(context) }

      let generation = try await session.generate(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        context: context
      )

      expectNoDifference(generation.engineGeneration.wasStopped, false)
      expectNoDifference(generation.engineGeneration.tokens, [])
      expectNoDifference(generation.engineGeneration.toolCalls.count, 1)
      expectNoDifference(generation.engineGeneration.toolCalls.first?.name, "send_email")
      expectNoDifference(generation.toolCalls.count, 1)
      withKnownIssue {
        assertSnapshot(of: generation.engineGeneration.parts, as: .dump, record: .all)
      }
      expectNoDifference(generation.engineGeneration.metrics.generationConfidence != nil, true)
      expectNoDifference(
        generation.engineGeneration.metrics.prefillTokensPerSecond.map { $0 > 0 },
        true
      )
      expectNoDifference(
        generation.engineGeneration.metrics.decodeTokensPerSecond.map { $0 > 0 },
        true
      )
      expectNoDifference(
        generation.engineGeneration.metrics.needle2PeakRAMMegabytes.map { $0 > 0 },
        true
      )
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Extracts Structured Data Through Needle 2`() async throws {
      let engine = Needle2Engine()
      let context = engine.context { DefinitionTool(Needle2Invoice.definition) }
      defer { try? engine.reset(context) }
      let task = try engine.generate(
        prompt: "Extract this invoice: Acme Corp, total $1,200.00, due 2026-09-01.",
        parameters: .default,
        context: context,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await task.value

      expectNoDifference(generation.wasStopped, false)
      expectNoDifference(generation.toolCalls.count, 1)
      expectNoDifference(
        generation.toolCalls.first?.name,
        Needle2Invoice.extractionToolDefinition.name
      )
      withKnownIssue {
        assertSnapshot(of: generation.parts, as: .dump, record: .all)
      }
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Drives Tool Loop Through Needle 2`() async throws {
      let engine = Needle2Engine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context { SendEmailTool() }
      defer { try? engine.reset(context) }

      let response = try await session.runLoop(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        context: context,
        maximumTurns: 2
      )
      expectNoDifference(response.steps.count, 2)
      expectNoDifference(response.steps.first?.generation.toolCalls.count, 1)
      expectNoDifference(response.steps.last?.generation.toolCalls.count, 0)
      expectNoDifference(response.terminationCause, .responded)
      withKnownIssue {
        assertSnapshot(of: response.steps.last?.generation.parts, as: .dump, record: .all)
      }
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Failed Tool Feeds Error Back Into Loop`() async throws {
      let engine = Needle2Engine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context { FailingSendEmailTool() }
      defer { try? engine.reset(context) }

      let response = try await session.runLoop(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        context: context,
        maximumTurns: 2
      )

      expectNoDifference(
        response.steps.first?.toolResponses,
        [.object(["error": .string("mailServerOffline")])]
      )
      expectNoDifference(response.steps.count, 2)
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Undecodable Tool Arguments Feed Error Back Into Loop`() async throws {
      let engine = Needle2Engine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context { MismatchedSchemaTool() }
      defer { try? engine.reset(context) }

      let response = try await session.runLoop(
        prompt: "Set the thermostat in the living room.",
        context: context,
        maximumTurns: 2
      )

      expectNoDifference(
        response.steps.first?.toolResponses,
        [.object(["error": .string("invalid arguments for tool: set_thermostat")])]
      )
      expectNoDifference(response.terminationCause, .responded)
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Existential Tools Encode Their Loop Responses`() async throws {
      let engine = Needle2Engine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context { SendEmailTool() }
      defer { try? engine.reset(context) }

      let response = try await session.runLoop(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        context: context,
        maximumTurns: 2
      )

      expectNoDifference(
        response.steps.first?.toolResponses,
        [.string("Sent email to blob@gmail.com")]
      )
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Rejects Turn Counts Needle 2 Cannot Support`() async throws {
      let engine = Needle2Engine()
      let session = EdgeToolsSession(engine: engine)
      let context = session.context { SendEmailTool() }
      defer { try? engine.reset(context) }

      await #expect(throws: Needle2LoopError.unsupportedTurnCount(9)) {
        try await session.runLoop(prompt: "Hello.", context: context, maximumTurns: 9)
      }
    }

    @Test
    @available(macOS 26, iOS 26, tvOS 26, watchOS 26, *)
    func `Stopped Generation Completes With Its Response`() async throws {
      let engine = Needle2Engine()
      let context = engine.context { SendEmailTool() }
      defer { try? engine.reset(context) }
      let emittedParts = Lock([EdgeToolsGenerationPart]())
      let didObserveResponse = Lock(false)
      withObservationTracking {
        _ = context.isResponding
      } onChange: {
        didObserveResponse.withLock { $0 = true }
      }
      let task = try engine.generate(
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
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
      expectNoDifference(generation.metrics.needle2PeakRAMMegabytes != nil, true)
    }
  }

  private enum Needle2ToolFailure: EdgeToolsDescribableError, Sendable {
    case mailServerOffline

    var edgeToolsErrorDescription: String {
      switch self {
      case .mailServerOffline: "mailServerOffline"
      }
    }
  }

  private struct FailingSendEmailTool: EdgeTool {
    typealias Input = SendEmailTool.Input

    let name = "sendEmail"
    let description = "Sends an email to a recipient with an email address."

    func invoke(input: Input) async throws -> String {
      throw Needle2ToolFailure.mailServerOffline
    }
  }

  @EdgeToolsGenerable
  private struct Needle2ThermostatRoom: Sendable {
    @EdgeToolsGuide(.description("The room to adjust."))
    var room: String
  }

  private struct MismatchedSchemaTool: EdgeTool {
    @EdgeToolsGenerable
    struct Input: Sendable {
      @EdgeToolsGuide(.description("The temperature to set."))
      var temperature: Int
    }

    let name = "setThermostat"
    let description = "Sets the thermostat temperature."

    var arguments: EdgeToolsGenerationSchema {
      Needle2ThermostatRoom.edgeToolsGenerationSchema
    }

    func invoke(input: Input) async throws -> String {
      "Set the thermostat to \(input.temperature)"
    }
  }

  @EdgeToolsGenerable
  private struct Needle2Invoice: Equatable, Sendable {
    @EdgeToolsGuide(.description("The vendor on the invoice."))
    var vendor: String

    @EdgeToolsGuide(.description("The total amount on the invoice."))
    var total: Double

    @EdgeToolsGuide(
      key: "due_date",
      .description("The due date on the invoice.")
    )
    var dueDate: String

    static var definition: EdgeToolDefinition { Self.extractionToolDefinition }
  }
#endif
