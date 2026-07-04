#if Sentencepiece
  import CustomDump
  import Foundation
  import Needle
  import Observation
  import Testing

  @Suite
  struct `NeedleSession tests` {
    @Test
    func `System Prompt Is Observable`() {
      let session = NeedleSession(engine: MockEngine())

      let didChange = Lock(false)
      withObservationTracking {
        _ = session.systemPrompt
      } onChange: {
        didChange.withLock { $0 = true }
      }

      session.systemPrompt = "new prompt"
      didChange.withLock { expectNoDifference($0, true) }
    }

    @Test
    func `Tokenize Forwards Tool Definitions And Prompt To The Engine`() async throws {
      let expectedTokens = (0..<6).map { NeedleToken(id: $0, stringValue: "t\($0)") }
      let tool = WeatherTool()
      let capturedPrompt = Lock<NeedlePrompt?>(nil)
      let engine = MockEngine { prompt in
        capturedPrompt.withLock { $0 = prompt }
        return expectedTokens
      }
      let session = NeedleSession(engine: engine, systemPrompt: "sys")

      let tokens = try await session.tokenize(
        tools: [tool],
        with: "hi",
        systemPromptOverride: nil
      )

      expectNoDifference(tokens, expectedTokens)

      let prompt = try #require(capturedPrompt.withLock { $0 })
      let expectedPrompt = NeedlePrompt(system: "sys", user: "hi", tools: [tool.definition])
      expectNoDifference(prompt, expectedPrompt)
    }

    @Test
    func `Final Generation Returns Successfully When No Errors Occur`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "Hello, world!".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")
      let generation = try await stream.finalGeneration

      expectNoDifference(generation.engineGeneration.tokens, tokens)
      expectNoDifference(generation.engineGeneration.wasStopped, false)
      expectNoDifference(generation.toolCalls.count, 0)
    }

    @Test
    func `Generate Returns Successfully With Parsed Tool Call`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let generation = try await session.generate(tools: [WeatherTool()], with: "weather?")

      expectNoDifference(generation.engineGeneration.tokens, toolTokens)
      expectNoDifference(generation.engineGeneration.wasStopped, false)
      expectNoDifference(generation.toolCalls.count, 1)
      expectNoDifference(generation.toolCalls[0].tool.name, "get_weather")
      let args = try #require(generation.toolCalls[0].input as? WeatherArgs)
      expectNoDifference(args.location, "Seoul")
    }

    @Test
    func `Generate Returns Successfully With Parsed Tool Call With Nested Objects And Arrays`()
      async throws
    {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall =
        #"<tool_call> [{"name":"plan_trip","arguments":{"destination":{"city":"Tokyo","country":"Japan"},"activities":[{"name":"Sushi","duration":2},{"name":"Temple","duration":3}],"tags":["food","culture"]}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let generation = try await session.generate(tools: [PlanTripTool()], with: "trip?")

      expectNoDifference(generation.toolCalls.count, 1)
      expectNoDifference(generation.toolCalls[0].tool.name, "plan_trip")
      let args = try #require(generation.toolCalls[0].input as? PlanTripArgs)
      expectNoDifference(args.destination.city, "Tokyo")
      expectNoDifference(args.destination.country, "Japan")
      expectNoDifference(args.activities.count, 2)
      expectNoDifference(args.activities[0].name, "Sushi")
      expectNoDifference(args.activities[0].duration, 2)
      expectNoDifference(args.activities[1].name, "Temple")
      expectNoDifference(args.activities[1].duration, 3)
      expectNoDifference(args.tags, ["food", "culture"])
    }

    @Test
    func `Generate Throws When Engine Errors`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let error = ToolError(message: "boom")
      let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])
      let session = NeedleSession(engine: engine)

      await #expect(throws: ToolError.self) {
        _ = try await session.generate(tools: [], with: "hi")
      }
    }

    @Test
    func `Generate Propagates Task Cancellation`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let firstToken = "a a".tokenize(using: tokenizer).first!
      let engine = MockEngine.live()
      engine.push(.token(firstToken))
      let session = NeedleSession(engine: engine)

      let task = Task {
        try await session.generate(tools: [], with: "hi")
      }

      await Task.yield()
      task.cancel()
      engine.push(.finish)
      engine.push(nil)

      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Final Generation Throws When Engine Errors`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let error = ToolError(message: "boom")
      let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      await #expect(throws: ToolError.self) {
        _ = try await stream.finalGeneration
      }
    }

    @Test
    func `Tools Stream Incrementally In Order As Mock Engine Emits Them`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCalls =
        #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#
      let toolTokens = rawToolCalls.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [WeatherTool()], with: "weather?")

      var collected = NeedleToolCallCollection()
      for try await call in stream {
        collected.append(call)
      }

      expectNoDifference(collected.count, 2)
      expectNoDifference(collected[0].tool.name, "get_weather")

      let firstArgs = try #require(collected[0].input as? WeatherArgs)
      expectNoDifference(firstArgs.location, "Seoul")
      expectNoDifference(collected[1].tool.name, "get_weather")

      let secondArgs = try #require(collected[1].input as? WeatherArgs)
      expectNoDifference(secondArgs.location, "Paris")
    }

    @Test
    func `Raw Tokens Are Buffered And Streamed Through The Tokens Sequence`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "abc".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      var collected = [NeedleToken]()
      for try await token in stream.tokens {
        collected.append(token)
      }

      expectNoDifference(collected, tokens)
    }

    @Test
    func `Stopping Stops Generation Within The Engine`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let firstToken = "a a".tokenize(using: tokenizer).first!
      let secondToken = "b b".tokenize(using: tokenizer).first!
      let engine = MockEngine.live()
      engine.push(.token(firstToken))
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      let generationTask = Task {
        try await stream.finalGeneration
      }

      try await Task.sleep(for: .milliseconds(50))
      stream.stop()
      engine.push(.token(secondToken))
      engine.push(.finish)
      engine.push(nil)

      let generation = try await generationTask.value
      expectNoDifference(generation.engineGeneration.wasStopped, true)
      expectNoDifference(generation.engineGeneration.tokens.count < 2, true)
    }

    @Test
    func `Stopping Before Generation Starts Returns An Empty Stopped Generation`() async throws {
      let engine = MockEngine.live()
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")
      stream.stop()

      let generation = try await stream.finalGeneration
      expectNoDifference(generation.engineGeneration.wasStopped, true)
      expectNoDifference(generation.toolCalls.count, 0)
      expectNoDifference(stream.isFinished, true)
    }

    @Test
    func `Stopping Before Generation Starts Updates Status Observation`() async throws {
      let engine = MockEngine.live()
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      let didChange = Lock(false)
      withObservationTracking {
        _ = stream.result
      } onChange: {
        didChange.withLock { $0 = true }
      }

      stream.stop()
      _ = try await stream.finalGeneration

      didChange.withLock { expectNoDifference($0, true) }
    }

    @Test
    func `Tools Are Parsed Incremental Without Waiting For Model Stop`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let trailing = "trailing".tokenize(using: tokenizer)
      let engine = MockEngine(
        script: toolTokens.map { .token($0) } + trailing.map { .token($0) } + [.finish]
      )
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [WeatherTool()], with: "weather?")

      var firstYielded: AnyNeedleToolCall?
      for try await call in stream {
        firstYielded = call
        break
      }

      expectNoDifference(firstYielded?.tool.name, "get_weather")
    }

    @Test
    func `Task Cancellation Propagates When Awaiting Final Generation`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let firstToken = "a a".tokenize(using: tokenizer).first!
      let engine = MockEngine.live()
      engine.push(.token(firstToken))
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      let task = Task {
        try await stream.finalGeneration
      }

      await Task.yield()
      task.cancel()
      engine.push(.finish)
      engine.push(nil)

      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Status Is Observable`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      let didChange = Lock(false)
      withObservationTracking {
        _ = stream.result
      } onChange: {
        didChange.withLock { $0 = true }
      }

      _ = try await stream.finalGeneration
      didChange.withLock { expectNoDifference($0, true) }
    }

    @Test
    func `Tool Calls Are Observable On Stream`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [WeatherTool()], with: "weather?")

      let didChange = Lock(false)
      withObservationTracking {
        _ = stream.toolCalls
      } onChange: {
        didChange.withLock { $0 = true }
      }

      _ = try await stream.finalGeneration
      didChange.withLock { expectNoDifference($0, true) }
    }

    @Test
    func `Active Streams Are Observable`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let didChange = Lock(false)
      withObservationTracking {
        _ = session.activeStreams
      } onChange: {
        didChange.withLock { $0 = true }
      }

      let stream = session.stream(tools: [], with: "hi")
      didChange.withLock { expectNoDifference($0, true) }

      _ = try await stream.finalGeneration
    }

    @Test
    func `Tool Call Parsed When Tool Name Differs From Snake Cased`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall =
        #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let generation = try await session.generate(
        tools: [CamelCaseWeatherTool()],
        with: "weather?"
      )

      expectNoDifference(generation.toolCalls.count, 1)
      expectNoDifference(generation.toolCalls[0].tool.name, "getWeather")
      let args = try #require(generation.toolCalls[0].input as? WeatherArgs)
      expectNoDifference(args.location, "Seoul")
    }

    @Test
    func `Active Streams Track Concurrent Streams And Remove On Finish`() async throws {
      let engine = MockEngine.live()
      let session = NeedleSession(engine: engine)

      let stream1 = session.stream(tools: [], with: "hi")
      try await Task.sleep(for: .milliseconds(50))

      let stream2 = session.stream(tools: [], with: "hi")

      let activeStreams = session.activeStreams
      expectNoDifference(activeStreams.count, 2)
      expectNoDifference(activeStreams.contains { $0 === stream1 }, true)
      expectNoDifference(activeStreams.contains { $0 === stream2 }, true)

      engine.push(.finish)
      engine.push(nil)
      _ = try await stream1.finalGeneration

      let activeStreamsAfterFirst = session.activeStreams
      expectNoDifference(activeStreamsAfterFirst.count, 1)
      expectNoDifference(activeStreamsAfterFirst.contains { $0 === stream1 }, false)
      expectNoDifference(activeStreamsAfterFirst.contains { $0 === stream2 }, true)

      engine.push(.finish)
      engine.push(nil)
      _ = try await stream2.finalGeneration

      expectNoDifference(session.activeStreams.count, 0)
    }

    @Test
    func `Status Is Generating While Engine Streams Tokens`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let firstToken = "a a".tokenize(using: tokenizer).first!
      let engine = MockEngine.live()
      engine.push(.token(firstToken))
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      try await Task.sleep(for: .milliseconds(50))
      expectNoDifference(stream.isGenerating, true)

      engine.push(.finish)
      engine.push(nil)
      _ = try await stream.finalGeneration
    }

    @Test
    func `Status Is Finished With Success When Engine Responds Successfully`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")
      let generation = try await stream.finalGeneration

      expectNoDifference(stream.isFinished, true)

      let resultGeneration = try #require(try? stream.result?.get())
      expectNoDifference(
        resultGeneration.engineGeneration.tokens,
        generation.engineGeneration.tokens
      )
      expectNoDifference(stream.isGenerating, false)
    }

    @Test
    func `Status Is Finished With Failure When Engine Errors`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let error = ToolError(message: "boom")
      let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      await #expect(throws: ToolError.self) {
        _ = try await stream.finalGeneration
      }
      expectNoDifference(stream.isFinished, true)

      let resultError = #expect(throws: ToolError.self) {
        _ = try stream.result?.get()
      }
      expectNoDifference(resultError?.message, error.message)
    }

    @Test
    func `Emitted Tools Are Idle When Should Invoke Tools Is False`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(
        tools: [WeatherTool()],
        with: "weather?",
        shouldInvokeTools: { _ in false }
      )

      var collected = NeedleToolCallCollection()
      for try await call in stream {
        collected.append(call)
      }

      expectNoDifference(collected.count, 1)
      expectNoDifference(collected[0].status.isIdle, true)
    }

    @Test
    func `Emitted Tools Are Idle Or Running When Should Invoke Tools Is True`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(
        tools: [BlockingWeatherTool()],
        with: "weather?",
        shouldInvokeTools: { _ in true }
      )

      var collected = NeedleToolCallCollection()
      for try await call in stream {
        collected.append(call)
      }

      expectNoDifference(collected.count, 1)
      let status = collected[0].status
      let isIdleOrRunning = status.isIdle || status.isRunning
      expectNoDifference(isIdleOrRunning, true)
    }

    @Test
    func `Generate Parses Tool Call When JSON Strings Contain Braces And Brackets`() async throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let rawToolCall =
        #"<tool_call> [{"name":"record_note","arguments":{"text":"literal}_and]_inside_string","title":"{draft}"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let generation = try await session.generate(tools: [RecordNoteTool()], with: "note?")

      expectNoDifference(generation.toolCalls.count, 1)
      guard generation.toolCalls.count == 1 else { return }
      let args = try #require(generation.toolCalls[0].input as? RecordNoteArgs)
      expectNoDifference(args.text, "literal}_and]_inside_string")
      expectNoDifference(args.title, "{draft}")
    }
  }

  // MARK: - WeatherArgs

  @NeedleGenerable
  private struct WeatherArgs: Equatable {
    var location: String
  }

  // MARK: - PlanTripArgs

  @NeedleGenerable
  private struct PlanTripArgs: Equatable {
    var destination: Destination
    var activities: [Activity]
    var tags: [String]
  }

  @NeedleGenerable
  private struct Destination: Equatable {
    var city: String
    var country: String
  }

  @NeedleGenerable
  private struct Activity: Equatable {
    var name: String
    var duration: Int
  }

  @NeedleGenerable
  private struct RecordNoteArgs: Equatable {
    var text: String
    var title: String
  }

  // MARK: - WeatherTool

  private struct WeatherTool: NeedleTool {
    typealias Input = WeatherArgs
    typealias Output = String

    let name = "get_weather"
    let description = "Gets the current weather for a location."

    func invoke(input: WeatherArgs) async throws -> sending String {
      "Sunny in \(input.location)"
    }
  }

  // MARK: - CamelCaseWeatherTool

  private struct CamelCaseWeatherTool: NeedleTool {
    typealias Input = WeatherArgs
    typealias Output = String

    let name = "getWeather"
    let description = "Gets the current weather for a location."

    func invoke(input: WeatherArgs) async throws -> sending String {
      "Sunny in \(input.location)"
    }
  }

  // MARK: - PlanTripTool

  private struct PlanTripTool: NeedleTool {
    typealias Input = PlanTripArgs
    typealias Output = String

    let name = "plan_trip"
    let description = "Plans a trip to a destination with activities."

    func invoke(input: PlanTripArgs) async throws -> sending String {
      "Trip to \(input.destination.city) with \(input.activities.count) activities"
    }
  }

  private struct RecordNoteTool: NeedleTool {
    typealias Input = RecordNoteArgs
    typealias Output = String

    let name = "record_note"
    let description = "Records a note."

    func invoke(input: RecordNoteArgs) async throws -> sending String {
      "\(input.title): \(input.text)"
    }
  }

  private final class BlockingWeatherTool: NeedleTool, Sendable {
    typealias Input = WeatherArgs
    typealias Output = String

    let name = "get_weather"
    let description = "Gets the current weather for a location."

    func invoke(input: WeatherArgs) async throws -> sending String {
      try await AsyncThrowingStream<Void, any Error> { _ in }.first { true }
      return "done"
    }
  }

  // MARK: - String + Tokenize

  extension String {
    fileprivate func tokenize(using tokenizer: NeedleSPTokenizer) -> [NeedleToken] {
      let strings = tokenizer.tokenize(text: self)
      return strings.enumerated()
        .compactMap { index, tokenString in
          guard let id = tokenizer.convertTokenToId(tokenString) else { return nil }
          if index == 0, tokenString.hasPrefix("▁") { return nil }
          return NeedleToken(id: id, stringValue: tokenString)
        }
    }
  }

  // MARK: - MockEngine

  private final class MockEngine: NeedleEngine, Sendable {
    fileprivate final class GenerationTask: NeedleEngineGenerationTask {
      private let task: Task<NeedleEngineGeneration, any Error>
      private let stopGeneration: @Sendable () -> Void

      init(
        task: sending Task<NeedleEngineGeneration, any Error>,
        stopGeneration: @escaping @Sendable () -> Void
      ) {
        self.task = task
        self.stopGeneration = stopGeneration
      }

      var value: NeedleEngineGeneration {
        get async throws {
          try await self.task.value
        }
      }

      func stop() {
        self.stopGeneration()
      }
    }

    struct GenerateParameters: NeedleEngineGenerateParameters {
      static let `default` = GenerateParameters()
    }

    enum Event: Sendable {
      case token(NeedleToken)
      case stop
      case error(any Error)
      case finish
    }

    private final class GenerationStorage: @unchecked Sendable {
      private let condition = NSCondition()
      private var queue = [Event?]()
      private var isStopped = false

      init(events: [Event?] = []) {
        self.queue = events
      }

      func push(_ event: Event?) {
        self.condition.lock()
        self.queue.append(event)
        self.condition.signal()
        self.condition.unlock()
      }

      func stop() {
        self.condition.lock()
        self.isStopped = true
        self.condition.broadcast()
        self.condition.unlock()
      }

      func resetStop() {
        self.condition.lock()
        self.isStopped = false
        self.condition.unlock()
      }

      func nextEvent() -> Event? {
        self.condition.lock()
        defer { self.condition.unlock() }

        while self.queue.isEmpty, !self.isStopped {
          self.condition.wait()
        }

        guard !self.isStopped else { return .stop }
        return self.queue.removeFirst()
      }
    }

    private final class Storage: @unchecked Sendable {
      private let condition = NSCondition()
      private var activeGenerations = [Int: GenerationStorage]()
      private var order = [Int]()
      private var nextID = 0
      private var queuedScripts = [[Event?]]()

      init(queuedScripts: [[Event?]] = []) {
        self.queuedScripts = queuedScripts
      }

      func makeGeneration() -> (Int, GenerationStorage) {
        self.condition.lock()
        defer { self.condition.unlock() }

        let id = self.nextID
        self.nextID += 1
        let script = self.queuedScripts.isEmpty ? [] : self.queuedScripts.removeFirst()
        let generation = GenerationStorage(events: script)
        self.activeGenerations[id] = generation
        self.order.append(id)
        return (id, generation)
      }

      func finishGeneration(id: Int) {
        self.condition.lock()
        self.activeGenerations.removeValue(forKey: id)
        self.order.removeAll { $0 == id }
        self.condition.unlock()
      }

      func push(_ event: Event?) {
        self.condition.lock()
        let generation = self.order.first.flatMap { self.activeGenerations[$0] }
        self.condition.unlock()
        generation?.push(event)
      }
    }

    private let storage: Storage
    private let _generateCallCount = Lock(0)
    private let tokenizeHandler: (@Sendable (NeedlePrompt) -> [NeedleToken])?
    private let _onGenerateStart = Lock<(@Sendable () -> Void)?>(nil)
    private let _onGenerateEnd = Lock<(@Sendable () -> Void)?>(nil)

    var onGenerateStart: (@Sendable () -> Void)? {
      get { self._onGenerateStart.withLock { $0 } }
      set { self._onGenerateStart.withLock { $0 = newValue } }
    }

    var onGenerateEnd: (@Sendable () -> Void)? {
      get { self._onGenerateEnd.withLock { $0 } }
      set { self._onGenerateEnd.withLock { $0 = newValue } }
    }

    var generateCallCount: Int {
      self._generateCallCount.withLock { $0 }
    }

    init() {
      let storage = Storage()
      self.storage = storage
      self.tokenizeHandler = nil
    }

    init(script: [Event]) {
      let storage = Storage(queuedScripts: [script.map { $0 } + [nil]])
      self.storage = storage
      self.tokenizeHandler = nil
    }

    init(tokenize: @escaping @Sendable (NeedlePrompt) -> [NeedleToken]) {
      let storage = Storage()
      self.storage = storage
      self.tokenizeHandler = tokenize
    }

    static func live() -> MockEngine {
      MockEngine()
    }

    func tokenize(prompt: NeedlePrompt) async throws -> [NeedleToken] {
      self.tokenizeHandler?(prompt) ?? []
    }

    func push(_ event: Event?) {
      self.storage.push(event)
    }

    func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void
    ) throws -> GenerationTask {
      self._generateCallCount.withLock { $0 += 1 }
      let (id, generationStorage) = self.storage.makeGeneration()
      let onStart = self._onGenerateStart.withLock { $0 }
      let onEnd = self._onGenerateEnd.withLock { $0 }
      let task = Task {
        onStart?()
        defer {
          onEnd?()
          self.storage.finishGeneration(id: id)
        }
        var emittedTokens = [NeedleToken]()
        var wasStopped = false
        var thrownError: (any Error)?

        try await withTaskCancellationHandler {
          while let event = generationStorage.nextEvent() {
            try Task.checkCancellation()
            switch event {
            case .token(let token):
              onToken(token)
              emittedTokens.append(token)
            case .stop:
              wasStopped = true
            case .error(let error):
              thrownError = error
            case .finish:
              break
            }
            if wasStopped || thrownError != nil { break }
          }
        } onCancel: {
          generationStorage.stop()
        }

        if let error = thrownError { throw error }
        return NeedleEngineGeneration(
          prefillMetrics: NeedlePrefillMetrics(tokens: 0, duration: .zero),
          decodeMetrics: NeedleDecodeMetrics(
            tokens: emittedTokens.count,
            duration: .zero,
            durationToFirstToken: .zero
          ),
          wasStopped: wasStopped,
          tokens: emittedTokens
        )
      }
      return GenerationTask(task: task) {
        generationStorage.stop()
      }
    }
  }

  // MARK: - GetWeatherTool

  private struct GetWeatherTool: NeedleTool {
    typealias Input = String
    typealias Output = String

    let name = "GetWeather"
    let description = ""

    func invoke(input: String) async throws -> sending String { "" }
  }

  // MARK: - Identical Snake Case Tool

  private struct GETWEATHERTOOL: NeedleTool {
    typealias Input = String
    typealias Output = String

    let name = "GETWEATHER"
    let description = ""

    func invoke(input: String) async throws -> sending String { "" }
  }

  // MARK: - Duplicate Tool Name Precondition

  @Suite
  struct `Duplicate Tool Name Precondition tests` {
    
    #if os(macOS) || os(linux) || os(windows)
      @Test
      func `Stream With Duplicate Tool Names Causes Precondition Failure`() async {
        await #expect(processExitsWith: .failure) {
          let session = NeedleSession(engine: MockEngine())
          _ = session.stream(
            tools: [CamelCaseWeatherTool(), GetWeatherTool()],
            with: "hi"
          )
        }
      }
    #endif

    @Test
    func `Duplicate Tool Name Error`() throws {
      let message = try #require(
        duplicateToolNameError(["getWeather", "GetWeather", "get_weather"])
      )
      #expect(message.contains("'getWeather'"))
      #expect(message.contains("'GetWeather'"))
      #expect(message.contains("'get_weather'"))
      #expect(message.contains("normalize to 'get_weather'"))
    }

    @Test
    func `No Error For Unique Names`() {
      let message = duplicateToolNameError(["getWeather", "planTrip"])
      #expect(message == nil)
    }

    @Test
    func `Duplicate Tool Name Error Detects Identical Names`() throws {
      let message = try #require(duplicateToolNameError(["get_weather", "get_weather"]))

      #expect(message.contains("'get_weather'"))
      #expect(message.contains("normalize to 'get_weather'"))
    }
  }
#endif
