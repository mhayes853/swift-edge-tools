import CustomDump
import EdgeTools
import Testing

struct `EdgeToolsAgent tests` {
  @Test
  func agentInvokesToolsInParallelAndPreservesResponseOrder() async throws {
    let tracker = ParallelInvocationTracker()
    let first = ParallelTool(name: "first", tracker: tracker)
    let second = ParallelTool(name: "second", tracker: tracker)
    let engine = AgentScriptEngine(
      generations: [
        .toolCalls([
          EdgeRawToolCall(name: first.name, arguments: .string("")),
          EdgeRawToolCall(name: second.name, arguments: .string(""))
        ]),
        .response(#""done""#)
      ]
    )
    let context = engine.context {
      first
      second
    }

    let result = try await engine.respond(
      to: .user("Run both tools."),
      as: String.self,
      context: context
    )

    expectNoDifference(result.output, "done")
    expectNoDifference(result.generations.count, 2)
    expectNoDifference(result.toolCalls.count, 2)
    expectNoDifference(
      context.prompts,
      [
        .user("Run both tools."),
        .tools([
          EdgeToolsTranscript.ToolMessage(name: first.name, response: .string("first:true")),
          EdgeToolsTranscript.ToolMessage(name: second.name, response: .string("second:true"))
        ])
      ]
    )
  }

  @Test
  func agentConfiguresConstraintForEachTurn() async throws {
    let engine = AgentScriptEngine(generations: [.response(#""done""#)])

    _ = try await engine.respond(
      to: .user("Respond."),
      as: String.self,
      context: engine.context(),
      constraint: { response, turn in
        .toolCallsOrResponse(response, toolCallRange: .exact(turn.index + 2))
      }
    )

    expectNoDifference(engine.constraints.map(\.toolCallRange), [.exact(2)])
  }

  @Test
  func `Typed Extraction Streams Partials And Finished Output`() async throws {
    let engine = AgentScriptEngine(generations: [.responseChunks(["\"hel", "lo\""])])
    let stream = engine.streamExtract(
      prompt: .user("Say hello."),
      as: String.self
    )

    var partials = [String]()
    var finished: String?
    for try await event in stream {
      switch event {
      case .partial(_, let partial):
        partials.append(String(partial))
      case .finish(.success(let result)):
        finished = result.output
      default: break
      }
    }

    expectNoDifference(partials, ["hel", "hello"])
    expectNoDifference(finished, "hello")
    let result = try await stream.finalResult
    expectNoDifference(result.output, "hello")
  }

  @Test
  func `Typed Extraction Publishes A Partial Before Generation Finishes`() async throws {
    let release = PartRelease()
    defer { release.resume() }
    let engine = AgentScriptEngine(
      generations: [.responseChunks(["\"hel", "lo\""])],
      pauseAfterFirstPart: release
    )
    let stream = engine.streamExtract(prompt: .user("Say hello."), as: String.self)
    var iterator = stream.makeAsyncIterator()
    var partial: String?
    while let event = try await iterator.next() {
      if case .partial(_, let value) = event {
        partial = String(value)
        break
      }
    }
    expectNoDifference(partial, "hel")

    release.resume()
    let result = try await stream.finalResult
    expectNoDifference(result.output, "hello")
  }

  @Test
  func `Typed Stream Closure Receives Finished Output`() async throws {
    let engine = AgentScriptEngine(generations: [.response("\"done\"")])
    let stream = engine.streamExtract(prompt: .user("Finish."), as: String.self)
    let finished = Lock<String?>(nil)

    let result = try await stream.consume { event in
      if case .finish(.success(let output)) = event {
        finished.withLock { $0 = output.output }
      }
    }

    expectNoDifference(result.output, "done")
    expectNoDifference(finished.withLock { $0 }, "done")
  }

  @Test
  func `Typed Response Streams Tool Calls And Finished Output`() async throws {
    let tool = ParallelTool(name: "lookup", tracker: ParallelInvocationTracker())
    let engine = AgentScriptEngine(
      generations: [
        .toolCalls([EdgeRawToolCall(name: tool.name, arguments: .string(""))]),
        .responseChunks(["\"do", "ne\""])
      ]
    )
    let context = engine.context { tool }
    let stream = engine.streamRespond(to: .user("Look it up."), as: String.self, context: context)

    var turns = [Int]()
    var toolNames = [String]()
    var partials = [String]()
    var finished: String?
    for try await event in stream {
      switch event {
      case .turnStarted(let turn):
        turns.append(turn)
      case .toolCall(_, let outcome):
        toolNames.append(outcome.name)
      case .partial(_, let partial):
        partials.append(String(partial))
      case .finish(.success(let result)):
        finished = result.output
      default: break
      }
    }

    let result = try await stream.finalResult
    expectNoDifference(turns, [0, 1])
    expectNoDifference(toolNames, ["lookup"])
    expectNoDifference(partials, ["do", "done"])
    expectNoDifference(finished, "done")
    expectNoDifference(result.generations.count, 2)
    expectNoDifference(result.toolCalls.count, 1)
  }

  @Test
  func `Typed Stream Shares Generation Callbacks And Tool State`() async throws {
    let tool = ParallelTool(name: "lookup", tracker: ParallelInvocationTracker())
    let engine = AgentScriptEngine(
      generations: [
        .toolCalls([EdgeRawToolCall(name: tool.name, arguments: .string(""))]),
        .responseChunks(["\"done\""])
      ]
    )
    let stream = engine.streamRespond(
      to: .user("Look it up."),
      as: String.self,
      context: engine.context { tool }
    )
    let parts = Lock([EdgeToolsGenerationPart]())
    let outcomes = Lock([String]())
    let calls = Lock([String]())
    let partSubscription = stream.onPart { part in parts.withLock { $0.append(part) } }
    let outcomeSubscription = stream.onToolCallOutcome { outcome in
      outcomes.withLock { $0.append(outcome.name) }
    }
    let callSubscription = stream.onToolCall { call in
      calls.withLock { $0.append(call.tool.name) }
    }
    defer {
      partSubscription.cancel()
      outcomeSubscription.cancel()
      callSubscription.cancel()
    }

    let result = try await stream.finalResult

    expectNoDifference(parts.withLock { $0.count }, 2)
    expectNoDifference(outcomes.withLock { $0 }, ["lookup"])
    expectNoDifference(calls.withLock { $0 }, ["lookup"])
    expectNoDifference(stream.toolCalls.count, 1)
    expectNoDifference(stream.toolCallOutcomes.count, 1)
    expectNoDifference(result.toolCallOutcomes.count, 1)
    expectNoDifference(stream.isFinished, true)
    expectNoDifference(try stream.result?.get().output, "done")
  }

  @Test
  func `Public Typed Continuation Publishes Custom Events`() async throws {
    let token = EdgeToolsToken(id: 1, stringValue: "hello")
    let stream = EdgeToolsTypedStream<String> { continuation in
      continuation.beginTurn(0)
      continuation.yield(token: token, turn: 0)
      continuation.yield(part: .text("hello"), turn: 0)
      continuation.yield(partial: "hello".streamPartialValue, turn: 0)
      return EdgeToolsTypedResult(
        output: "hello",
        generations: [],
        toolCalls: EdgeToolCallCollection()
      )
    }

    let events = Lock([String]())
    let result = try await stream.consume { event in
      switch event {
      case .turnStarted: events.withLock { $0.append("start") }
      case .token: events.withLock { $0.append("token") }
      case .part: events.withLock { $0.append("part") }
      case .partial: events.withLock { $0.append("partial") }
      case .finish(.success): events.withLock { $0.append("finish") }
      default: break
      }
    }

    expectNoDifference(result.output, "hello")
    expectNoDifference(events.withLock { $0 }, ["start", "token", "part", "partial", "finish"])
  }

  @Test
  func `Typed Sequences Replay Tokens And The Finish Event`() async throws {
    let token = EdgeToolsToken(id: 1, stringValue: "hello")
    let stream = EdgeToolsTypedStream<String> { continuation in
      continuation.yield(token: token, turn: 0)
      return EdgeToolsTypedResult(
        output: "hello",
        generations: [],
        toolCalls: EdgeToolCallCollection()
      )
    }
    _ = try await stream.finalResult

    var tokens = [EdgeToolsToken]()
    for try await value in stream.tokens {
      tokens.append(value)
    }
    var finished = false
    for await event in stream.events {
      if case .finish(.success) = event {
        finished = true
      }
    }

    expectNoDifference(tokens, [token])
    expectNoDifference(finished, true)
  }

  @Test
  func `Public Typed Continuation Receives Stop Requests`() async throws {
    let release = PartRelease()
    let stopped = Lock(false)
    let stream = EdgeToolsTypedStream<String> { continuation in
      continuation.onStop {
        stopped.withLock { $0 = true }
        release.resume()
      }
      await release.wait()
      return EdgeToolsTypedResult(
        output: "unreachable",
        generations: [],
        toolCalls: EdgeToolCallCollection()
      )
    }

    stream.stop()
    await #expect(throws: CancellationError.self) {
      _ = try await stream.finalResult
    }
    expectNoDifference(stopped.withLock { $0 }, true)
  }

  @Test
  func `Typed Stream Rejects An Incomplete Output`() async throws {
    let engine = AgentScriptEngine(generations: [.response("{}")])
    let stream = engine.streamExtract(
      prompt: .user("Return an object."),
      as: RequiredPayload.self
    )

    await #expect(throws: EdgeToolsTypedStreamError.self) {
      _ = try await stream.finalResult
    }
    await #expect(throws: EdgeToolsTypedStreamError.self) {
      for try await event in stream {
        _ = event
      }
    }
  }

  @Test
  func `Typed Extraction Converts A Macro Generated Partial`() async throws {
    let engine = AgentScriptEngine(
      generations: [.responseChunks(["{\"value\":\"he", "llo\"}"])]
    )
    let stream = engine.streamExtract(
      prompt: .user("Return an object."),
      as: RequiredPayload.self
    )

    let result = try await stream.finalResult
    expectNoDifference(result.output.value, "hello")

    var partials = [String]()
    for try await event in stream {
      if case .partial(_, let partial) = event {
        partials.append(partial.value.map(String.init) ?? "")
      }
    }
    expectNoDifference(partials, ["he", "hello"])
  }

  @Test
  func `Stopping A Typed Stream Cancels Its Result`() async throws {
    let release = PartRelease()
    defer { release.resume() }
    let engine = AgentScriptEngine(
      generations: [.responseChunks(["\"hel", "lo\""])],
      pauseAfterFirstPart: release
    )
    let stream = engine.streamExtract(prompt: .user("Say hello."), as: String.self)
    var iterator = stream.makeAsyncIterator()
    _ = try await iterator.next()
    _ = try await iterator.next()

    stream.stop()
    release.resume()
    await #expect(throws: CancellationError.self) {
      _ = try await stream.finalResult
    }
  }
}

@EdgeToolsGenerable
private struct RequiredPayload: Sendable {
  var value: String
}

private struct AgentTurnConstraint:
  EdgeToolsTurnGenerationConstraint, EdgeToolsSchemaGenerationConstraint
{
  typealias Grammar = Int
  typealias Context = Void

  var toolCallRange: GrammarToolCallRange?

  static func toolCallsOrResponse<Response: EdgeToolsGenerable>(
    _ response: Response.Type,
    toolCallRange: GrammarToolCallRange
  ) -> Self {
    Self(toolCallRange: toolCallRange)
  }

  static func schema(_ schema: EdgeToolsGenerationSchema) -> Self {
    Self(toolCallRange: nil)
  }

  func grammar(toolCallGrammar: consuming Int?, context: Void) throws -> Int {
    0
  }
}

private final class AgentScriptEngine: EdgeToolsEngine {
  final class Context: EdgeToolsEngineContext {
    private let _prompts = Lock([EdgeToolsTranscript.Prompt]())
    let tools: [any EdgeTool]
    let isResponding = false

    init(tools: [any EdgeTool]) {
      self.tools = tools
    }

    var prompts: [EdgeToolsTranscript.Prompt] {
      self._prompts.withLock { $0 }
    }

    func append(_ prompt: EdgeToolsTranscript.Prompt) {
      self._prompts.withLock { $0.append(prompt) }
    }
  }

  struct GenerateParameters: EdgeToolsConstrainedGenerateParameters {
    static let `default` = Self(constraint: AgentTurnConstraint(toolCallRange: nil))

    var constraint: AgentTurnConstraint
    var maxTokens: Int? { nil }
  }

  typealias Prompt = EdgeToolsTranscript.Prompt
  typealias GenerationTask = AnyGenerationTask

  private let generations: Lock<[EdgeToolsEngineGeneration]>
  private let _constraints = Lock<[AgentTurnConstraint]>([])
  private let pauseAfterFirstPart: PartRelease?

  var constraints: [AgentTurnConstraint] {
    self._constraints.withLock { $0 }
  }

  init(generations: [EdgeToolsEngineGeneration], pauseAfterFirstPart: PartRelease? = nil) {
    self.generations = Lock(generations)
    self.pauseAfterFirstPart = pauseAfterFirstPart
  }

  func context() -> Context {
    Context(tools: [])
  }

  func context(tools: [any EdgeTool]) -> Context {
    Context(tools: tools)
  }

  func context(_ parameters: Void, tools: [any EdgeTool]) -> Context {
    Context(tools: tools)
  }

  func generationTask(
    prompt: Prompt,
    parameters: sending GenerateParameters,
    context: Context,
    continuation: sending EdgeToolsGenerationStream.Continuation
  ) throws -> AnyGenerationTask {
    context.append(prompt)
    self._constraints.withLock { $0.append(parameters.constraint) }
    let generation = self.generations.withLock { $0.removeFirst() }
    return AnyGenerationTask { _ in
      for (index, part) in generation.parts.enumerated() {
        continuation.yield(part: part)
        if index == 0 {
          await self.pauseAfterFirstPart?.wait()
        }
      }
      return generation
    }
  }
}

private final class PartRelease: Sendable {
  private struct State {
    var continuation: UnsafeContinuation<Void, Never>?
    var hasResumed = false
  }

  private let state = Lock(State())

  func wait() async {
    await withUnsafeContinuation { continuation in
      let shouldResume = self.state.withLock { state in
        guard !state.hasResumed else { return true }
        state.continuation = continuation
        return false
      }
      if shouldResume {
        continuation.resume()
      }
    }
  }

  func resume() {
    let continuation = self.state.withLock { state in
      state.hasResumed = true
      defer { state.continuation = nil }
      return state.continuation
    }
    continuation?.resume()
  }
}

extension EdgeToolsEngineGeneration {
  fileprivate static func toolCalls(_ calls: [EdgeRawToolCall]) -> Self {
    Self(
      wasStopped: false,
      tokens: [],
      response: "",
      parts: calls.map(EdgeToolsGenerationPart.toolCall)
    )
  }

  fileprivate static func response(_ response: String) -> Self {
    Self(
      wasStopped: false,
      tokens: [],
      response: response,
      parts: [.text(response)]
    )
  }

  fileprivate static func responseChunks(_ chunks: [String]) -> Self {
    Self(
      wasStopped: false,
      tokens: [],
      response: chunks.joined(),
      parts: chunks.map(EdgeToolsGenerationPart.text)
    )
  }
}

private struct ParallelTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name: String
  let tracker: ParallelInvocationTracker

  var description: String { "Records whether both tools ran concurrently." }

  func invoke(input: String) async throws -> String {
    self.tracker.markStarted()
    try await Task.sleep(for: .milliseconds(25))
    return "\(self.name):\(self.tracker.hasBothStarted)"
  }
}

private final class ParallelInvocationTracker: Sendable {
  private let count = Lock(0)

  var hasBothStarted: Bool {
    self.count.withLock { $0 == 2 }
  }

  func markStarted() {
    self.count.withLock { $0 += 1 }
  }
}
