import EdgeToolsCore
import StreamParsing
import _Concurrency

// MARK: - Streaming

extension EdgeToolsEngine {
  public func stream(
    prompt: Prompt,
    context: Context,
    parameters: sending GenerateParameters = .default,
    textEmission: EdgeToolsTextEmission = .everyToken,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) -> EdgeToolsGenerationStream {
    let stream = EdgeToolsGenerationStream(
      tools: context.tools,
      textEmission: textEmission,
      shouldInvokeTools: shouldInvokeTools
    )
    stream.start(
      engine: self,
      prompt: prompt,
      context: context,
      parameters: parameters
    )
    return stream
  }

  @concurrent
  public func generate(
    prompt: Prompt,
    context: Context,
    parameters: sending GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) async throws -> EdgeToolsGeneration {
    try await self.stream(
      prompt: prompt,
      context: context,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    .finalGeneration
  }
}

// MARK: - Extraction

extension EdgeToolsEngine
where
  GenerateParameters: EdgeToolsConstrainedGenerateParameters,
  GenerateParameters.Constraint: EdgeToolsSchemaGenerationConstraint
{
  @concurrent
  public func extract<Response: EdgeToolsGenerable>(
    prompt: Prompt,
    as type: Response.Type,
    parameters: sending GenerateParameters = .default
  ) async throws -> Response {
    var parameters = parameters
    parameters.constraint = .schema(type.edgeToolsGenerationSchema)
    let task = try self.generationTask(
      prompt: prompt,
      parameters: parameters,
      context: self.context(),
      continuation: .discarding
    )
    let generation = try await task.value
    return try Response(edgeToolsValue: EdgeToolsValue(json: generation.text))
  }
}

// MARK: - Streaming Extraction

extension EdgeToolsEngine
where
  GenerateParameters: EdgeToolsConstrainedGenerateParameters,
  GenerateParameters.Constraint: EdgeToolsSchemaGenerationConstraint
{
  /// Streams partial values from one schema-constrained generation.
  public func streamExtract<Output>(
    prompt: Prompt,
    as type: Output.Type,
    context: Context,
    parameters: sending GenerateParameters = .default,
    textEmission: EdgeToolsTextEmission = .everyToken
  ) -> EdgeToolsTypedStream<Output>
  where Output: EdgeToolsGenerable & StreamParseable & Sendable, Output.Partial: Sendable {
    var parameters = parameters
    parameters.constraint = .schema(type.edgeToolsGenerationSchema)
    let generation = self.stream(
      prompt: prompt,
      context: context,
      parameters: parameters,
      textEmission: textEmission,
      shouldInvokeTools: { _ in false }
    )
    return EdgeToolsTypedStream { continuation in
      continuation.beginTurn(0)
      let (completed, parser) = try await continuation.generation(generation, turn: 0)
      continuation.finishTurn(completed, turn: 0)
      return EdgeToolsTypedResult(
        output: try parser.complete(fallbackText: completed.text),
        generations: [completed],
        toolCalls: completed.toolCalls
      )
    }
  }

  /// Streams partial values using a new context.
  public func streamExtract<Output>(
    prompt: Prompt,
    as type: Output.Type,
    parameters: sending GenerateParameters = .default,
    textEmission: EdgeToolsTextEmission = .everyToken
  ) -> EdgeToolsTypedStream<Output>
  where Output: EdgeToolsGenerable & StreamParseable & Sendable, Output.Partial: Sendable {
    self.streamExtract(
      prompt: prompt,
      as: type,
      context: self.context(),
      parameters: parameters,
      textEmission: textEmission
    )
  }
}
