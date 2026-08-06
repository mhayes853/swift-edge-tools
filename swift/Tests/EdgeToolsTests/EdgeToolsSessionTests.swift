import CustomDump
import EdgeTools
import Foundation
import Observation
import Testing

@Suite(.serialized)
struct `EdgeToolsSession tests` {
  @Test
  func `Tokenize Forwards Prompt To The Engine`() async throws {
    let expectedTokens = (0..<6).map { EdgeToolsToken(id: $0, stringValue: "t\($0)") }
    let tool = WeatherTool()
    let prompt = NeedlePrompt(system: "sys", user: "hi")
    let captured = Lock<(NeedlePrompt, [EdgeToolDefinition])?>(nil)
    let engine = MockEngine { prompt, tools in
      captured.withLock { $0 = (prompt, tools) }
      return expectedTokens
    }
    let session = EdgeToolsSession(engine: engine) { tool }

    let tokens = try await session.tokenize(prompt: prompt)

    expectNoDifference(tokens, expectedTokens)

    let capturedValues = try #require(captured.withLock { $0 })
    expectNoDifference(capturedValues.0.system, "sys")
    expectNoDifference(capturedValues.0.user, "hi")
    expectNoDifference(capturedValues.1, [tool.definition])
  }

  @Test
  func `Tools Are Observable`() {
    let session = EdgeToolsSession(engine: MockEngine())
    let didChange = Lock(false)
    withObservationTracking {
      _ = session.tools
    } onChange: {
      didChange.withLock { $0 = true }
    }

    session.tools = [EdgeToolsSessionTool(EchoTool())]

    didChange.withLock { expectNoDifference($0, true) }
  }

  @Test
  func `Final Generation Returns Successfully When No Errors Occur`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "Hello, world!".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))
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
    let session = EdgeToolsSession(engine: engine) { WeatherTool() }

    let generation = try await session.generate(prompt: .test(user: "weather?"))

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
  func `Engine Generation Returns Raw Tool Calls`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"unknown","arguments":{"value":1}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])

    let task = try engine.generate(
      prompt: .test(user: "call it"),
      tools: [.sendEmail],
      parameters: .default,
      channel: EdgeToolsGenerationChannel()
    )
    let generation = try await task.value

    expectNoDifference(
      generation.toolCalls,
      [EdgeRawToolCall(name: "unknown", arguments: ["value": 1])]
    )
  }

  @Test
  func `Generate Throws When Engine Errors`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let error = ToolError(message: "boom")
    let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])
    let session = EdgeToolsSession(engine: engine)

    await #expect(throws: ToolError.self) {
      _ = try await session.generate(prompt: .test(user: "hi"))
    }
  }

  @Test
  func `Generate Propagates Task Cancellation`() async throws {
    let tokenizer = try testTokenizer()
    let firstToken = "a a".tokenize(using: tokenizer).first!
    let engine = MockEngine.live()
    engine.push(.token(firstToken))
    let session = EdgeToolsSession(engine: engine)

    let task = Task {
      try await session.generate(prompt: .test(user: "hi"))
    }

    while engine.generateCallCount == 0 { await Task.yield() }
    task.cancel()
    engine.push(.finish)
    engine.push(nil)

    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }
  }

  @Test
  func `Final Generation Throws When Engine Errors`() async throws {
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let error = ToolError(message: "boom")
    let engine = MockEngine(script: tokens.map { .token($0) } + [.error(error)])
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine) { WeatherTool() }

    let stream = session.stream(prompt: .test(user: "weather?"))

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
    let session = EdgeToolsSession(engine: engine) { WeatherTool() }

    let stream = session.stream(prompt: .test(user: "weather?"))

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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))
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
    let session = EdgeToolsSession(engine: engine)
    let stream = session.stream(prompt: ReentrantMockEngine.Prompt())
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
      case .toolCall: break
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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine)
    let streamedTokens = try await self.tokensFromCompletedStream(session: session)

    var collected = [EdgeToolsToken]()
    for try await token in streamedTokens {
      collected.append(token)
    }

    expectNoDifference(collected, tokens)
  }

  private func tokensFromCompletedStream(
    session: EdgeToolsSession<MockEngine>
  ) async throws -> EdgeToolsSessionTokens {
    let stream = session.stream(prompt: .test(user: "hi"))
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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))
    stream.stop()

    let generation = try await stream.finalGeneration
    expectNoDifference(generation.engineGeneration.wasStopped, true)
    expectNoDifference(generation.toolCalls.count, 0)
    expectNoDifference(stream.isFinished, true)
  }

  @Test
  func `Stopping Before Generation Starts Updates Status Observation`() async throws {
    let engine = MockEngine.live()
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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

extension `EdgeToolsSession tests` {
  @Test
  func `Tools Are Parsed Incremental Without Waiting For Model Stop`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let trailing = "trailing".tokenize(using: tokenizer)
    let engine = MockEngine(
      script: toolTokens.map { .token($0) } + trailing.map { .token($0) } + [.finish]
    )
    let session = EdgeToolsSession(engine: engine) { WeatherTool() }

    let stream = session.stream(prompt: .test(user: "weather?"))

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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine) { WeatherTool() }

    let stream = session.stream(prompt: .test(user: "weather?"))

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
    let tokenizer = try testTokenizer()
    let tokens = "hi".tokenize(using: tokenizer)
    let engine = MockEngine(script: tokens.map { .token($0) } + [.finish])
    let session = EdgeToolsSession(engine: engine)

    let didChange = Lock(false)
    withObservationTracking {
      _ = session.activeStreams
    } onChange: {
      didChange.withLock { $0 = true }
    }

    let stream = session.stream(prompt: .test(user: "hi"))
    didChange.withLock { expectNoDifference($0, true) }

    _ = try await stream.finalGeneration
  }

  @Test
  func `Tool Call Parsed When Tool Name Differs From Snake Cased`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall =
      #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine(script: toolTokens.map { .token($0) } + [.finish])
    let session = EdgeToolsSession(engine: engine) { CamelCaseWeatherTool() }

    let generation = try await session.generate(prompt: .test(user: "weather?"))

    expectNoDifference(generation.toolCalls.count, 1)
    expectNoDifference(generation.toolCalls[0].tool.name, "getWeather")
    expectNoDifference(generation.toolCalls[0].rawValue.name, "getWeather")
    let args = try #require(generation.toolCalls[0].input as? WeatherArgs)
    expectNoDifference(args.location, "Seoul")
  }

  @Test
  func `Active Generation Uses Tools Snapshot`() async throws {
    let tokenizer = try testTokenizer()
    let rawToolCall = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#
    let toolTokens = rawToolCall.tokenize(using: tokenizer)
    let engine = MockEngine.live()
    let didStart = Lock(false)
    engine.onGenerateStart = { didStart.withLock { $0 = true } }
    let weatherTool = WeatherTool()
    let session = EdgeToolsSession(engine: engine) { weatherTool }

    let stream = session.stream(prompt: .test(user: "weather?"))
    while !didStart.withLock({ $0 }) { await Task.yield() }
    session.tools = [EdgeToolsSessionTool(EchoTool())]
    for token in toolTokens { engine.push(.token(token)) }
    engine.push(.finish)
    engine.push(nil)

    let generation = try await stream.finalGeneration
    let nextStream = session.stream(prompt: .test(user: "echo"))
    while engine.generateCallCount < 2 { await Task.yield() }
    engine.push(.finish)
    engine.push(nil)
    _ = try await nextStream.finalGeneration

    expectNoDifference(
      engine.generationTools,
      [[weatherTool.definition], [EchoTool().definition]]
    )
    expectNoDifference(generation.toolCalls.count, 1)
    expectNoDifference(generation.toolCalls[0].tool.name, weatherTool.name)
    expectNoDifference(session.tools.map { $0.name }, ["echo"])
  }

  @Test
  func `Generation Decodes Its Response`() async throws {
    let response = #"{"name":"Ada"}"#
    let engine = MockEngine(
      script: [.token(EdgeToolsToken(id: 0, stringValue: response)), .finish]
    )
    let session = EdgeToolsSession(engine: engine)

    let generation = try await session.generate(prompt: .test(user: "hi"))
    let value = try generation.decoded(as: EdgeToolsValue.self)

    expectNoDifference(value, ["name": "Ada"])
  }

  @Test
  func `Stream Decodes Its Final Response`() async throws {
    let response = #"{"name":"Ada"}"#
    let engine = MockEngine(
      script: [.token(EdgeToolsToken(id: 0, stringValue: response)), .finish]
    )
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))
    let value = try await stream.decodedResponse(as: EdgeToolsValue.self)

    expectNoDifference(value, ["name": "Ada"])
  }

  @Test
  func `Active Streams Track Concurrent Streams And Remove On Finish`() async throws {
    let engine = MockEngine.live()
    let session = EdgeToolsSession(engine: engine)

    let stream1 = session.stream(prompt: .test(user: "hi"))
    try await Task.sleep(for: .milliseconds(50))

    let stream2 = session.stream(prompt: .test(user: "hi"))

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

    expectNoDifference(session.activeStreams.isEmpty, true)
  }

  @Test
  func `Status Is Generating While Engine Streams Tokens`() async throws {
    let tokenizer = try testTokenizer()
    let firstToken = "a a".tokenize(using: tokenizer).first!
    let engine = MockEngine.live()
    engine.push(.token(firstToken))
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))
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
    let session = EdgeToolsSession(engine: engine)

    let stream = session.stream(prompt: .test(user: "hi"))

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
    let session = EdgeToolsSession(engine: engine) { WeatherTool() }

    let stream = session.stream(
      prompt: .test(user: "weather?"),
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
    let session = EdgeToolsSession(engine: engine) { BlockingWeatherTool() }

    let stream = session.stream(
      prompt: .test(user: "weather?"),
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
        let session = EdgeToolsSession(engine: MockEngine()) {
          CamelCaseWeatherTool()
          GetWeatherTool()
        }
        _ = session.stream(prompt: .test(user: "hi"))
      }
    }
  #endif

  @Test
  func `Duplicate Tool Name Error`() throws {
    let message = try #require(
      duplicateToolNameError(["getWeather", "GetWeather", "get_weather"])
    )
    expectNoDifference(message.contains("'getWeather'"), true)
    expectNoDifference(message.contains("'GetWeather'"), true)
    expectNoDifference(message.contains("'get_weather'"), true)
    expectNoDifference(message.contains("normalize to 'get_weather'"), true)
  }

  @Test
  func `No Error For Unique Names`() {
    let message = duplicateToolNameError(["getWeather", "planTrip"])
    expectNoDifference(message, nil)
  }

  @Test
  func `Duplicate Tool Name Error Detects Identical Names`() throws {
    let message = try #require(duplicateToolNameError(["get_weather", "get_weather"]))

    expectNoDifference(message.contains("'get_weather'"), true)
    expectNoDifference(message.contains("normalize to 'get_weather'"), true)
  }
}

extension NeedlePrompt {
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
    let strings = tokenizer.tokenize(text: self)
    return strings.enumerated()
      .compactMap { index, tokenString in
        guard let id = tokenizer.convertTokenToId(tokenString) else { return nil }
        if index == 0, tokenString.hasPrefix("▁") { return nil }
        return EdgeToolsToken(id: id, stringValue: tokenString)
      }
  }
}

// MARK: - GetWeatherTool

private struct GetWeatherTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "GetWeather"
  let description = ""

  func invoke(input: String) async throws -> sending String { "" }
}

// MARK: - Reentrant Mock Engine

private final class ReentrantMockEngine: EdgeToolsEngine {
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

  func tokenize(prompt: Prompt, tools: [EdgeToolDefinition]) async throws -> [EdgeToolsToken] {
    []
  }

  func generate(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    parameters: GenerateParameters,
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

  func invoke(input: String) async throws -> sending String { "" }
}
