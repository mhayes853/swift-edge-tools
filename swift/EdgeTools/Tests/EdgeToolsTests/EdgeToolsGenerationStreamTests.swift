import CustomDump
import EdgeTools
import Foundation
import Observation
import Testing

@Suite(.serialized)
struct `EdgeToolsGenerationStream tests` {
  @Test
  func `Tokenize Forwards Prompt To The Engine`() async throws {
    let expectedTokens = (0..<6).map { EdgeToolsToken(id: $0, stringValue: "t\($0)") }
    let tool = WeatherTool()
    let prompt = TestPrompt(system: "sys", user: "hi")
    let captured = Lock<(TestPrompt, [EdgeToolDefinition])?>(nil)
    let engine = MockEngine { prompt, tools in
      captured.withLock { $0 = (prompt, tools) }
      return expectedTokens
    }
    let context = engine.context { tool }

    let tokens = try await engine.tokenize(prompt: prompt, context: context)

    expectNoDifference(tokens, expectedTokens)

    let capturedValues = try #require(captured.withLock { $0 })
    expectNoDifference(capturedValues.0.system, "sys")
    expectNoDifference(capturedValues.0.user, "hi")
    expectNoDifference(capturedValues.1, [tool.definition])
  }

  @Test
  func `Final Generation Returns Successfully When No Errors Occur`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "Hello, world!".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())
    let generation = try await stream.finalGeneration

    expectNoDifference(generation.engineGeneration.tokens, tokens)
    expectNoDifference(generation.engineGeneration.wasStopped, false)
    expectNoDifference(generation.toolCalls.count, 0)
  }

  @Test
  func `Generate Returns Successfully With Parsed Tool Call`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { WeatherTool() }

    let generation = try await engine.generate(prompt: .test(user: "weather?"), context: context)

    expectNoDifference(generation.engineGeneration.tokens, toolTokens)
    expectNoDifference(generation.engineGeneration.wasStopped, false)
    expectNoDifference(
      generation.engineGeneration.toolCalls,
      [EdgeRawToolCall(name: "get_weather", arguments: ["location": "Seoul"])]
    )
    expectNoDifference(generation.toolCalls.count, 1)
    expectNoDifference(generation.toolCalls[0].tool.name, "get_weather")
    let args = try #require(generation.toolCalls[0].input as? WeatherArgs)
    expectNoDifference(args.location, "Seoul")
    expectNoDifference(generation.toolCalls[0].rawValue.arguments, ["location": "Seoul"])
  }

  @Test
  func `Extracts A Typed Value With Structured Generation`() async throws {
    let tokenizer = try testTokenizer()
    let responseTokens = #"{"location":"Seoul"}"#.tokenize(using: tokenizer)
    let engine = MockEngine(script: responseTokens.map { .token($0) } + [.finish])

    let value = try await engine.extract(
      prompt: .test(user: "weather?"),
      as: WeatherArgs.self
    )

    expectNoDifference(value.location, "Seoul")
    expectNoDifference(engine.generationTools, [[]])
    expectNoDifference(
      engine.generationConstraints,
      [MockEngine.GenerationConstraint(schema: WeatherArgs.edgeToolsGenerationSchema)]
    )
  }

  @Test
  func `Extraction Throws Without A Structured Response`() async {
    let engine = MockEngine(script: [.finish])

    await #expect(throws: EdgeToolsError.self) {
      try await engine.extract(
        prompt: .test(user: "not structured"),
        as: WeatherArgs.self
      )
    }
  }

  @Test
  func `Engine Generation Returns Raw Tool Calls`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"unknown","arguments":{"value":1}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])

    let task = try engine.generationTask(
      prompt: .test(user: "call it"),
      parameters: .default,
      context: engine.context(tools: [DefinitionTool(.sendEmail)]),
      channel: EdgeToolsGenerationChannel()
    )
    let generation = try await task.value

    expectNoDifference(
      generation.toolCalls,
      [EdgeRawToolCall(name: "unknown", arguments: ["value": 1])]
    )
  }

  @Test
  func `Final Generation Throws When Engine Errors`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let error = ToolError(message: "boom")
    let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

    await #expect(throws: ToolError.self) {
      _ = try await stream.finalGeneration
    }
    let response = try #require(stream.response)
    #expect(throws: ToolError.self) {
      try response.get()
    }
  }

  @Test
  func `Tools Stream Incrementally In Order As Mock Engine Emits Them`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCalls =
      #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#
    let toolTokens = rawToolCalls.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { WeatherTool() }

    let stream = engine.stream(prompt: .test(user: "weather?"), context: context)

    var collected = EdgeToolCallCollection()
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
  func `Subscriptions Receive Tokens And Tool Calls As They Are Emitted`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { WeatherTool() }

    let stream = engine.stream(prompt: .test(user: "weather?"), context: context)

    let tokens = Lock([EdgeToolsToken]())
    let names = Lock([String]())
    let tokenSubscription = stream.onToken { token in tokens.withLock { $0.append(token) } }
    let callSubscription = stream.onToolCall { call in
      names.withLock { $0.append(call.tool.name) }
    }
    defer {
      tokenSubscription.cancel()
      callSubscription.cancel()
    }

    _ = try await stream.finalGeneration

    tokens.withLock { expectNoDifference($0, toolTokens) }
    names.withLock { expectNoDifference($0, ["get_weather"]) }
  }

  @Test
  func `Subscribing After The Stream Finishes Replays Its Tokens And Tool Calls`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "abc".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())
    _ = try await stream.finalGeneration

    let replayed = Lock([EdgeToolsToken]())
    let subscription = stream.onToken { token in replayed.withLock { $0.append(token) } }
    defer { subscription.cancel() }

    replayed.withLock { expectNoDifference($0, tokens) }
  }

  @Test
  func `Subscription Replay Precedes Reentrant Token Delivery`() async throws {
    let firstToken = EdgeToolsToken(id: 1, stringValue: "first")
    let secondToken = EdgeToolsToken(id: 2, stringValue: "second")
    let engine = ReentrantMockEngine()
    let stream = engine.stream(
      prompt: ReentrantMockEngine.Prompt(),
      context: engine.context()
    )
    await engine.waitUntilReady()

    engine.emit(firstToken)

    let events = Lock([String]())
    let replayFinished = Lock(false)
    let tokenWasDeliveredDuringReplay = Lock(false)
    let subscription = stream.onEvent { event in
      switch event {
      case .token(let token):
        events.withLock { $0.append(token.stringValue) }
        if token == firstToken {
          engine.emit(secondToken)
          replayFinished.withLock { $0 = true }
        } else if token == secondToken {
          tokenWasDeliveredDuringReplay.withLock { $0 = !replayFinished.withLock { $0 } }
        }
      case .part: break
      case .finish: events.withLock { $0.append("finish") }
      @unknown default: break
      }
    }
    defer { subscription.cancel() }

    engine.finish()
    _ = try await stream.finalGeneration

    expectNoDifference(tokenWasDeliveredDuringReplay.withLock { $0 }, false)
    events.withLock { expectNoDifference($0, ["first", "second", "finish"]) }
  }

  @Test
  func `Cancelled Subscriptions Stop Receiving Tokens`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "abc".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

    let collected = Lock([EdgeToolsToken]())
    let subscription = stream.onToken { token in collected.withLock { $0.append(token) } }
    subscription.cancel()

    _ = try await stream.finalGeneration

    collected.withLock { expectNoDifference($0.isEmpty, true) }
  }

  @Test
  func `Raw Tokens Are Buffered And Streamed Through The Tokens Sequence`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "abc".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

    var collected = [EdgeToolsToken]()
    for try await token in stream.tokens {
      collected.append(token)
    }

    expectNoDifference(collected, tokens)
  }

  @Test
  func `Token Sequence Retains Its Completed Stream While Replaying Tokens`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "abc".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
    let streamedTokens = try await self.tokensFromCompletedStream(engine: engine)

    var collected = [EdgeToolsToken]()
    for try await token in streamedTokens {
      collected.append(token)
    }

    expectNoDifference(collected, tokens)
  }

  private func tokensFromCompletedStream(
    engine: MockEngine
  ) async throws -> EdgeToolsGenerationTokens {
    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())
    _ = try await stream.finalGeneration
    return stream.tokens
  }

  @Test
  func `Stopping Stops Generation Within The Engine`() async throws {
    let tokenizer = try testTokenizer()
    let firstToken = "a a".tokenize(using: tokenizer).first!
    let secondToken = "b b".tokenize(using: tokenizer).first!
    let engine = MockEngine.live()
    engine.push(.token(firstToken))

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

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

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())
    stream.stop()

    let generation = try await stream.finalGeneration
    expectNoDifference(generation.engineGeneration.wasStopped, true)
    expectNoDifference(generation.toolCalls.count, 0)
    expectNoDifference(stream.isFinished, true)
  }

  @Test
  func `Stopping Before Generation Starts Updates Status Observation`() async throws {
    let engine = MockEngine.live()

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

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
}

extension `EdgeToolsGenerationStream tests` {
  @Test
  func `Tools Are Parsed Incremental Without Waiting For Model Stop`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let trailing = "trailing".tokenize(using: tokenizer)
    let engine = MockEngine(
      script: toolTokens.map { .token($0) } + trailing.map { .token($0) } + [.finish]
    )
    let context = engine.context { WeatherTool() }

    let stream = engine.stream(prompt: .test(user: "weather?"), context: context)

    var firstYielded: AnyEdgeToolCall?
    for try await call in stream {
      firstYielded = call
      break
    }

    expectNoDifference(firstYielded?.tool.name, "get_weather")
  }

  @Test
  func `Task Cancellation Propagates When Awaiting Final Generation`() async throws {
    let tokenizer = try testTokenizer()
    let firstToken = "a a".tokenize(using: tokenizer).first!
    let engine = MockEngine.live()
    engine.push(.token(firstToken))

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

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
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

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
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { WeatherTool() }

    let stream = engine.stream(prompt: .test(user: "weather?"), context: context)

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
  func `Tool Call Parsed When Tool Name Differs From Snake Cased`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall =
      #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { CamelCaseWeatherTool() }

    let generation = try await engine.generate(prompt: .test(user: "weather?"), context: context)

    expectNoDifference(generation.toolCalls.count, 1)
    expectNoDifference(generation.toolCalls[0].tool.name, "getWeather")
    expectNoDifference(generation.toolCalls[0].rawValue.name, "getWeather")
    let args = try #require(generation.toolCalls[0].input as? WeatherArgs)
    expectNoDifference(args.location, "Seoul")
  }

  @Test
  func `Different Contexts Use Different Tool Sets`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(scripts: [toolTokens.map { .token($0) } + [.finish], [.finish]])
    let weatherTool = WeatherTool()
    let weatherContext = engine.context { weatherTool }
    let echoContext = engine.context { EchoTool() }

    let generation = try await engine.generate(prompt: .test(user: "weather?"), context: weatherContext)
    _ = try await engine.generate(prompt: .test(user: "echo"), context: echoContext)

    expectNoDifference(
      engine.generationTools,
      [[weatherTool.definition], [EchoTool().definition]]
    )
    expectNoDifference(generation.toolCalls.count, 1)
    expectNoDifference(generation.toolCalls[0].tool.name, weatherTool.name)
    expectNoDifference(weatherContext.tools.map(\.name), [weatherTool.name])
    expectNoDifference(echoContext.tools.map(\.name), ["echo"])
  }

  @Test
  func `Generation Decodes Its Response`() async throws {
    let response = #"{"name":"Ada"}"#
    let engine = MockEngine(
      script: [.token(EdgeToolsToken(id: 0, stringValue: response)), .finish]
    )

    let generation = try await engine.generate(
      prompt: .test(user: "hi"),
      context: engine.context()
    )
    let value = try generation.decoded(as: EdgeToolsValue.self)

    expectNoDifference(value, ["name": "Ada"])
  }

  @Test
  func `Status Is Generating While Engine Streams Tokens`() async throws {
    let tokenizer = try testTokenizer()
    let firstToken = "a a".tokenize(using: tokenizer).first!
    let engine = MockEngine.live()
    engine.push(.token(firstToken))

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

    try await Task.sleep(for: .milliseconds(50))
    expectNoDifference(stream.isGenerating, true)

    engine.push(.finish)
    engine.push(nil)
    _ = try await stream.finalGeneration
  }

  @Test
  func `Status Is Finished With Success When Engine Responds Successfully`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())
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
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let error = ToolError(message: "boom")
    let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])

    let stream = engine.stream(prompt: .test(user: "hi"), context: engine.context())

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
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { WeatherTool() }

    let stream = engine.stream(
      prompt: .test(user: "weather?"),
      context: context,
      shouldInvokeTools: { _ in false }
    )

    var collected = EdgeToolCallCollection()
    for try await call in stream {
      collected.append(call)
    }

    expectNoDifference(collected.count, 1)
    expectNoDifference(collected[0].status.isIdle, true)
  }

  @Test
  func `Emitted Tools Are Idle Or Running When Should Invoke Tools Is True`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let context = engine.context { BlockingWeatherTool() }

    let stream = engine.stream(
      prompt: .test(user: "weather?"),
      context: context,
      shouldInvokeTools: { _ in true }
    )

    var collected = EdgeToolCallCollection()
    for try await call in stream {
      collected.append(call)
    }

    expectNoDifference(collected.count, 1)
    let status = collected[0].status
    let isIdleOrRunning = status.isIdle || status.isRunning
    expectNoDifference(isIdleOrRunning, true)
  }

  #if os(macOS) || os(linux) || os(windows)
    @Test
    func `Stream With Duplicate Tool Names Causes Precondition Failure`() async {
      await #expect(processExitsWith: .failure) {
        let engine = MockEngine()
        let context = engine.context {
          CamelCaseWeatherTool()
          GetWeatherTool()
        }
        _ = engine.stream(prompt: .test(user: "hi"), context: context)
      }
    }
  #endif

}

extension TestPrompt {
  fileprivate static func test(user: String) -> Self {
    Self(system: "", user: user)
  }

}

// MARK: - WeatherArgs

@EdgeToolsGenerable
private struct WeatherArgs: Equatable {
  var location: String
}

// MARK: - WeatherTool

private struct WeatherTool: EdgeTool {
  typealias Input = WeatherArgs
  typealias Output = String

  let name = "get_weather"
  let description = "Gets the current weather for a location."

  func invoke(input: WeatherArgs) async throws -> sending String {
    "Sunny in \(input.location)"
  }
}

// MARK: - CamelCaseWeatherTool

private struct CamelCaseWeatherTool: EdgeTool {
  typealias Input = WeatherArgs
  typealias Output = String

  let name = "getWeather"
  let description = "Gets the current weather for a location."

  func invoke(input: WeatherArgs) async throws -> sending String {
    "Sunny in \(input.location)"
  }
}

private final class BlockingWeatherTool: EdgeTool, Sendable {
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
  fileprivate func tokenize(using tokenizer: some EdgeToolsTokenizer)
    -> [EdgeToolsToken]
  {
    tokenizer.encode(text: self).enumerated()
      .compactMap { index, token in
        if index == 0, token.stringValue.hasPrefix("▁") {
          return nil
        }
        return token
      }
  }
}

// MARK: - GetWeatherTool

private struct GetWeatherTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "GetWeather"
  let description = ""

  func invoke(input: String) async throws -> sending String {
    ""
  }
}

// MARK: - Reentrant Mock Engine

private final class ReentrantMockEngine: EdgeToolsEngine, EdgeToolsTokenizingEngine {
  final class Context: EdgeToolsEngineContext {
    let tools: [any EdgeTool]
    let isResponding = false

    init(tools: [any EdgeTool]) {
      self.tools = tools
    }
  }

  struct Prompt: Sendable {}

  struct GenerateParameters: EdgeToolsEngineGenerateParameters {
    static let `default` = GenerateParameters()

    var maxTokens: Int? { nil }
  }

  final class GenerationTask: EdgeToolsEngineGenerationTask {
    private let stream: AsyncStream<EdgeToolsEngineGeneration>
    private let continuation: AsyncStream<EdgeToolsEngineGeneration>.Continuation

    init() {
      let stream = AsyncStream<EdgeToolsEngineGeneration>.makeStream()
      self.stream = stream.stream
      self.continuation = stream.continuation
    }

    var value: EdgeToolsEngineGeneration {
      get async throws {
        for await generation in self.stream { return generation }
        return .empty
      }
    }

    func finish() {
      self.continuation.yield(.empty)
      self.continuation.finish()
    }

    func stop() {
      self.continuation.finish()
    }
  }

  private let channel = Lock<ReentrantChannelStorage?>(nil)
  private let task = GenerationTask()
  private let ready = AsyncStream<Void>.makeStream()

  func context(_ parameters: Void, tools: [any EdgeTool]) -> Context {
    Context(tools: tools)
  }

  func tokenize(
    prompt: Prompt,
    context: Context
  ) async throws -> [EdgeToolsToken] {
    []
  }

  func generationTask(
    prompt: Prompt,
    parameters: GenerateParameters,
    context: Context,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> GenerationTask {
    let channelStorage = ReentrantChannelStorage(channel: channel)
    self.channel.withLock { $0 = channelStorage }
    self.ready.continuation.yield()
    self.ready.continuation.finish()
    return self.task
  }

  func emit(_ token: EdgeToolsToken) {
    self.channel.withLock { $0?.emit(token: token) }
  }

  func finish() {
    self.channel.withLock { storage in
      storage?.finish()
      storage = nil
    }
    self.task.finish()
  }

  func waitUntilReady() async {
    for await _ in self.ready.stream { return }
  }
}

// MARK: - Reentrant Channel Storage

private final class ReentrantChannelStorage: Sendable {
  private let channel: Lock<EdgeToolsGenerationChannel?>

  init(channel: sending EdgeToolsGenerationChannel) {
    self.channel = Lock(channel)
  }

  func emit(token: EdgeToolsToken) {
    self.channel.withLock { $0?.emit(token: token) }
  }

  func finish() {
    self.channel.withLock { $0 = nil }
  }
}

// MARK: - Identical Snake Case Tool

private struct GETWEATHERTOOL: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "GETWEATHER"
  let description = ""

  func invoke(input: String) async throws -> sending String {
    ""
  }
}
