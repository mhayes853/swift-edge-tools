#if SwiftNeedleSentencepiece
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
    func `Final Generation Returns Successfully When No Errors Occur`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let tokens = "Hello, world!".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")
      let generation = try await stream.finalGeneration

      expectNoDifference(generation.engineGeneration.tokens, tokens)
      expectNoDifference(generation.engineGeneration.wasStoped, false)
      expectNoDifference(generation.toolCalls.count, 0)
    }

    @Test
    func `Generate Returns Successfully With Parsed Tool Call`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let generation = try await session.generate(tools: [WeatherTool()], with: "weather?")

      expectNoDifference(generation.engineGeneration.tokens, toolTokens)
      expectNoDifference(generation.engineGeneration.wasStoped, false)
      expectNoDifference(generation.toolCalls.count, 1)
      expectNoDifference(generation.toolCalls[0].tool.name, "get_weather")
      let args = try #require(generation.toolCalls[0].input as? WeatherArgs)
      expectNoDifference(args.location, "Seoul")
    }

    @Test
    func `Generate Throws When Engine Errors`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      expectNoDifference(generation.engineGeneration.wasStoped, true)
      expectNoDifference(generation.engineGeneration.tokens, [firstToken])
    }

    @Test
    func `Tools Are Parsed Incremental Without Waiting For Model Stop`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
    func `Result Is Observable`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
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
    func `Result Is Nil While Engine Is Still Streaming`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let firstToken = "a a".tokenize(using: tokenizer).first!
      let engine = MockEngine.live()
      engine.push(.token(firstToken))
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      try await Task.sleep(for: .milliseconds(50))
      let isNil = stream.result == nil
      expectNoDifference(isNil, true)
      expectNoDifference(stream.isResponding, true)

      engine.push(.finish)
      engine.push(nil)
      _ = try await stream.finalGeneration
    }

    @Test
    func `Result Is Success When Engine Responds Successfully`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")
      let generation = try await stream.finalGeneration

      guard case .success(let resultGeneration) = stream.result else {
        Issue.record("Expected result to be .success, got \(String(describing: stream.result))")
        return
      }
      expectNoDifference(
        resultGeneration.engineGeneration.tokens,
        generation.engineGeneration.tokens
      )
      expectNoDifference(stream.isResponding, false)
    }

    @Test
    func `Result Is Failure When Engine Errors`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let tokens = "hi".tokenize(using: tokenizer)
      let error = ToolError(message: "boom")
      let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(tools: [], with: "hi")

      await #expect(throws: ToolError.self) {
        _ = try await stream.finalGeneration
      }
      guard case .failure(let resultError) = stream.result else {
        Issue.record("Expected result to be .failure, got \(String(describing: stream.result))")
        return
      }
      expectNoDifference((resultError as? ToolError)?.message, error.message)
    }

    @Test
    func `Emitted Tools Are Idle When Should Invoke Tools Is False`() async throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(
        tools: [WeatherTool()],
        with: "weather?",
        shouldInvokeTools: false
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
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
      let toolTokens = rawToolCall.tokenize(using: tokenizer)
      let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
      let session = NeedleSession(engine: engine)

      let stream = session.stream(
        tools: [BlockingWeatherTool()],
        with: "weather?",
        shouldInvokeTools: true
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
  }

  // MARK: - WeatherArgs

  @NeedleGenerable
  private struct WeatherArgs: Equatable {
    var location: String
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
    fileprivate func tokenize(using tokenizer: NeedleSPTokenizingModel) -> [NeedleToken] {
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
    struct GenerateParameters: NeedleEngineGenerateParameters {
      static let `default` = GenerateParameters()
    }

    enum Event: Sendable {
      case token(NeedleToken)
      case stop
      case error(any Error)
      case finish
    }

    private final class Storage: @unchecked Sendable {
      private let condition = NSCondition()
      private var queue = [Event?]()
      private var isStopped = false

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

    private let storage: Storage
    let stopper: NeedleEngineStopper

    init() {
      let storage = Storage()
      self.storage = storage
      self.stopper = NeedleEngineStopper { storage.stop() }
    }

    init(script: [Event]) {
      let storage = Storage()
      self.storage = storage
      self.stopper = NeedleEngineStopper { storage.stop() }
      for event in script {
        storage.push(event)
      }
      storage.push(nil)
    }

    static func live() -> MockEngine {
      MockEngine()
    }

    func push(_ event: Event?) {
      self.storage.push(event)
    }

    func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParameters,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      self.storage.resetStop()
      var emittedTokens: [NeedleToken] = []
      var wasStopped = false
      var thrownError: (any Error)?

      while let event = self.storage.nextEvent() {
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

    func reset() {}
  }
#endif
