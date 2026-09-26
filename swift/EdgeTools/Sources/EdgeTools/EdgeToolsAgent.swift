import EdgeToolsCore
import OrderedCollections
import StreamParsing
import _Concurrency

// MARK: - EdgeToolsAgentTurn

public struct EdgeToolsAgentTurn<Context: EdgeToolsEngineContext>: Sendable {
  public let index: Int
  public let context: Context
  public let prompt: EdgeToolsTranscript.Prompt
}

// MARK: - EdgeToolsAgentResult

public struct EdgeToolsAgentResult<Result: Sendable>: Sendable {
  public let output: Result
  public let generations: [EdgeToolsGeneration]
  public let toolCalls: EdgeToolCallCollection
}

// MARK: - EdgeToolsAgentError

public enum EdgeToolsAgentError: Error, Sendable {
  case maximumTurnsExceeded(Int)
}

// MARK: - Agent Generation

extension EdgeToolsEngine
where
  Prompt == EdgeToolsTranscript.Prompt,
  GenerateParameters: EdgeToolsConstrainedGenerateParameters,
  GenerateParameters.Constraint: EdgeToolsTurnGenerationConstraint
{
  @concurrent
  public func respond<Result: EdgeToolsGenerable>(
    to initialPrompt: Prompt,
    as type: Result.Type,
    context: Context,
    maximumTurns: Int? = nil,
    parameters: @escaping @Sendable (EdgeToolsAgentTurn<Context>) -> GenerateParameters = {
      _ in .default
    },
    constraint:
      @escaping @Sendable (
        Result.Type,
        EdgeToolsAgentTurn<Context>
      ) -> GenerateParameters.Constraint = {
        type,
        _ in .toolCallsOrResponse(type, toolCallRange: .unbounded(minimum: 1))
      }
  ) async throws -> EdgeToolsAgentResult<Result> {
    var prompt = initialPrompt
    var generations = [EdgeToolsGeneration]()
    var toolCalls = EdgeToolCallCollection()

    var index = 0
    while maximumTurns.map({ index < $0 }) ?? true {
      let turn = EdgeToolsAgentTurn(index: index, context: context, prompt: prompt)
      var generationParameters = parameters(turn)
      generationParameters.constraint = constraint(Result.self, turn)
      let generation = try await self.generate(
        prompt: prompt,
        context: context,
        parameters: generationParameters
      )
      generations.append(generation)
      toolCalls.append(contentsOf: generation.toolCalls)

      guard !generation.toolCalls.isEmpty else {
        return EdgeToolsAgentResult(
          output: try generation.decoded(as: Result.self),
          generations: generations,
          toolCalls: toolCalls
        )
      }

      prompt = .tools(await agentToolResponses(for: generation.toolCallOutcomes))
      index += 1
    }

    throw EdgeToolsAgentError.maximumTurnsExceeded(maximumTurns!)
  }
}

// MARK: - Streaming Agent Generation

extension EdgeToolsEngine
where
  Prompt == EdgeToolsTranscript.Prompt,
  GenerateParameters: EdgeToolsConstrainedGenerateParameters,
  GenerateParameters.Constraint: EdgeToolsTurnGenerationConstraint
{
  /// Streams partial output and tool activity across an agent response.
  public func streamRespond<Output>(
    to initialPrompt: Prompt,
    as type: Output.Type,
    context: Context,
    maximumTurns: Int? = nil,
    textEmission: EdgeToolsTextEmission = .everyToken,
    parameters: @escaping @Sendable (EdgeToolsAgentTurn<Context>) -> GenerateParameters = {
      _ in .default
    },
    constraint:
      @escaping @Sendable (
        Output.Type,
        EdgeToolsAgentTurn<Context>
      ) -> GenerateParameters.Constraint = {
        type,
        _ in .toolCallsOrResponse(type, toolCallRange: .unbounded(minimum: 1))
      }
  ) -> EdgeToolsTypedStream<Output>
  where Output: EdgeToolsGenerable & StreamParseable & Sendable, Output.Partial: Sendable {
    EdgeToolsTypedStream { continuation in
      var prompt = initialPrompt
      var generations = [EdgeToolsGeneration]()
      var toolCalls = EdgeToolCallCollection()
      var index = 0

      while maximumTurns.map({ index < $0 }) ?? true {
        try Task.checkCancellation()
        continuation.beginTurn(index)
        let turn = EdgeToolsAgentTurn(index: index, context: context, prompt: prompt)
        var generationParameters = parameters(turn)
        generationParameters.constraint = constraint(type, turn)
        let raw = self.stream(
          prompt: prompt,
          context: context,
          parameters: generationParameters,
          textEmission: textEmission
        )
        let (generation, parser) = try await continuation.generation(raw, turn: index)
        generations.append(generation)
        toolCalls.append(contentsOf: generation.toolCalls)
        continuation.finishTurn(generation, turn: index)

        guard !generation.toolCalls.isEmpty else {
          return EdgeToolsTypedResult(
            output: try parser.complete(fallbackText: generation.text),
            generations: generations,
            toolCalls: toolCalls
          )
        }

        prompt = .tools(await agentToolResponses(for: generation.toolCallOutcomes))
        index += 1
      }

      throw EdgeToolsAgentError.maximumTurnsExceeded(maximumTurns!)
    }
  }
}

// MARK: - Tool Responses

func agentToolResponses(
  for outcomes: [EdgeToolCallOutcome]
) async -> [EdgeToolsTranscript.ToolMessage] {
  await withTaskGroup(of: (Int, EdgeToolsTranscript.ToolMessage).self) { group in
    for (index, outcome) in outcomes.enumerated() {
      group.addTask {
        let response = await outcome.response()
        return (index, EdgeToolsTranscript.ToolMessage(name: outcome.name, response: response))
      }
    }

    var responses = [EdgeToolsTranscript.ToolMessage?](
      repeating: nil,
      count: outcomes.count
    )
    for await (index, response) in group {
      responses[index] = response
    }
    return responses.compactMap { $0 }
  }
}

extension EdgeToolCallOutcome {
  fileprivate func response() async -> EdgeToolsValue {
    switch self {
    case .unknownTool:
      return edgeToolErrorResponse("unknown tool: \(self.name)")
    case .invalidArguments:
      return edgeToolErrorResponse("invalid arguments for tool: \(self.name)")
    case .resolved(let call):
      do {
        return try await call.outputValue
      } catch {
        return edgeToolErrorResponse(edgeToolErrorDescription(error))
      }
    }
  }
}

private func edgeToolErrorResponse(_ message: String) -> EdgeToolsValue {
  .object(["error": .string(message)])
}

private func edgeToolErrorDescription(_ error: any Error) -> String {
  (error as? EdgeToolsError)?.message ?? "unknown error"
}
