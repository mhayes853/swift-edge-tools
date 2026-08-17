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
          EdgeRawToolCall(name: second.name, arguments: .string("")),
        ]),
        .response(#""done""#),
      ]
    )
    let session = EdgeToolsSession(engine: engine) {
      first
      second
    }
    let context = session.context()

    let result = try await session.respond(
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
          .tool(name: first.name, response: .string("first:true")),
          .tool(name: second.name, response: .string("second:true")),
        ]),
      ]
    )
  }

  @Test
  func agentConfiguresConstraintForEachTurn() async throws {
    let engine = AgentScriptEngine(generations: [.response(#""done""#)])
    let session = EdgeToolsSession(engine: engine)

    _ = try await session.respond(
      to: .user("Respond."),
      as: String.self,
      constraint: { response, turn in
        .toolCallsOrResponse(response, toolCallRange: .exact(turn.index + 2))
      }
    )

    expectNoDifference(engine.constraints.map(\.toolCallRange), [.exact(2)])
  }
}

private struct AgentTurnConstraint: EdgeToolsTurnGenerationConstraint {
  typealias Grammar = Int
  typealias Context = Void

  var toolCallRange: GrammarToolCallRange?

  static func toolCallsOrResponse<Response: EdgeToolsGenerable>(
    _ response: Response.Type,
    toolCallRange: GrammarToolCallRange
  ) -> Self {
    Self(toolCallRange: toolCallRange)
  }

  func grammar(toolCallGrammar: consuming Int?, context: Void) throws -> Int {
    0
  }
}

private final class AgentScriptEngine: EdgeToolsEngine {
  final class Context: Identifiable, Sendable {
    private let _prompts = Lock([EdgeToolsTranscript.Prompt]())

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

  var constraints: [AgentTurnConstraint] {
    self._constraints.withLock { $0 }
  }

  init(generations: [EdgeToolsEngineGeneration]) {
    self.generations = Lock(generations)
  }

  func context(_ parameters: Void) -> Context {
    Context()
  }

  func generate(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    parameters: sending GenerateParameters,
    context: Context,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> AnyGenerationTask {
    context.append(prompt)
    self._constraints.withLock { $0.append(parameters.constraint) }
    let generation = self.generations.withLock { $0.removeFirst() }
    return AnyGenerationTask { _ in
      for part in generation.parts {
        channel.emit(part: part)
      }
      return generation
    }
  }
}

private extension EdgeToolsEngineGeneration {
  static func toolCalls(_ calls: [EdgeRawToolCall]) -> Self {
    Self(
      prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
      decodeMetrics: EdgeToolsDecodeMetrics(
        tokens: 0,
        duration: .zero,
        durationToFirstToken: .zero
      ),
      wasStopped: false,
      tokens: [],
      response: "",
      parts: calls.map(EdgeToolsGenerationPart.toolCall)
    )
  }

  static func response(_ response: String) -> Self {
    Self(
      prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
      decodeMetrics: EdgeToolsDecodeMetrics(
        tokens: 0,
        duration: .zero,
        durationToFirstToken: .zero
      ),
      wasStopped: false,
      tokens: [],
      response: response,
      parts: [.text(response)]
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
