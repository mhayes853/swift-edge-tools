#if Needle2
  import EdgeToolsCore
  import _Concurrency

  // MARK: - Needle2LoopResponse

  public struct Needle2LoopResponse: Sendable {
    public let steps: [Needle2LoopStep]
    public let terminationCause: Needle2LoopTerminationCause

    public init(
      steps: [Needle2LoopStep],
      terminationCause: Needle2LoopTerminationCause
    ) {
      self.steps = steps
      self.terminationCause = terminationCause
    }
  }

  // MARK: - Needle2LoopStep

  public struct Needle2LoopStep: Sendable {
    public let generation: EdgeToolsSessionGeneration
    public let toolResponses: [EdgeToolsValue]

    public init(
      generation: EdgeToolsSessionGeneration,
      toolResponses: [EdgeToolsValue]
    ) {
      self.generation = generation
      self.toolResponses = toolResponses
    }
  }

  // MARK: - Needle2LoopTerminationCause

  public enum Needle2LoopTerminationCause: Hashable, Sendable {
    case responded
    case refused
    case noToolCalls
    case maximumTurnsReached
  }

  // MARK: - Needle2LoopError

  public enum Needle2LoopError: Error, Hashable, Sendable {
    case unsupportedTurnCount(Int)
  }

  // MARK: - Extraction

  extension EdgeToolsSession where Engine: Needle2SessionEngine {
    @concurrent
    public func extract<Response: EdgeToolsGenerable>(
      prompt: Engine.Prompt,
      as type: Response.Type,
      parameters: sending Engine.GenerateParameters = .default
    ) async throws -> Response {
      let tool = Needle2ExtractionTool<Response>()
      let context = self.engine.context(tools: [tool])
      do {
        let task = try self.engine.generate(
          prompt: prompt,
          parameters: parameters,
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
        let generation = try await task.value
        guard let call = generation.toolCalls.first(where: { $0.name == tool.name }) else {
          throw Needle2Error(
            code: .invalidResponse,
            message: "Needle 2 did not produce the required extraction tool call."
          )
        }
        let response = try type.init(edgeToolsValue: call.arguments)
        try await self.engine.reset(context)
        return response
      } catch {
        try? await self.engine.reset(context)
        throw error
      }
    }
  }

  // MARK: - Loop

  extension EdgeToolsSession where Engine: Needle2SessionEngine {
    @concurrent
    public func runLoop(
      prompt: String,
      context: Engine.Context,
      maximumTurns: Int = 8,
      parameters: @escaping @Sendable (Int) -> Engine.GenerateParameters = { _ in .default },
      shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
    ) async throws -> Needle2LoopResponse {
      guard (1...8).contains(maximumTurns) else {
        throw Needle2LoopError.unsupportedTurnCount(maximumTurns)
      }

      var prompt = Needle2Prompt.user(prompt)
      var steps = [Needle2LoopStep]()

      for turn in 0..<maximumTurns {
        let generation = try await self.generate(
          prompt: prompt,
          context: context,
          parameters: parameters(turn),
          shouldInvokeTools: shouldInvokeTools
        )
        guard generation.engineGeneration.metrics.needle2ResponseType == "call",
          !generation.toolCallOutcomes.isEmpty
        else {
          steps.append(Needle2LoopStep(generation: generation, toolResponses: []))
          return Needle2LoopResponse(
            steps: steps,
            terminationCause: needle2LoopTerminationCause(for: generation)
          )
        }

        let toolResponses = await agentToolResponses(for: generation.toolCallOutcomes)
          .map(\.response)
        steps.append(Needle2LoopStep(generation: generation, toolResponses: toolResponses))
        guard turn < maximumTurns - 1 else {
          return Needle2LoopResponse(
            steps: steps,
            terminationCause: .maximumTurnsReached
          )
        }
        prompt = .toolResponses(toolResponses)
      }

      fatalError("Needle 2 loop ended without a termination cause.")
    }
  }

  // MARK: - Helpers

  private struct Needle2ExtractionTool<Response: EdgeToolsGenerable>: EdgeTool {
    let name = "extract"

    var description: String {
      Response.edgeToolsGenerationSchema.objectValue?[.description]?.string
        ?? "Extract structured data from the input."
    }

    func invoke(input: Response) async -> Response {
      input
    }
  }

  private func needle2LoopTerminationCause(
    for generation: EdgeToolsSessionGeneration
  ) -> Needle2LoopTerminationCause {
    switch generation.engineGeneration.metrics.needle2ResponseType {
    case "respond": .responded
    case "refuse": .refused
    default: .noToolCalls
    }
  }
#endif
