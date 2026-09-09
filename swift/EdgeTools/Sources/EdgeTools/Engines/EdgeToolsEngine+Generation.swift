import EdgeToolsCore
import _Concurrency

// MARK: - Streaming

extension EdgeToolsEngine {
  public func stream(
    prompt: Prompt,
    context: Context,
    parameters: sending GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) -> EdgeToolsGenerationStream {
    let stream = EdgeToolsGenerationStream(
      tools: context.tools,
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
      channel: EdgeToolsGenerationChannel()
    )
    let generation = try await task.value
    return try Response(edgeToolsValue: EdgeToolsValue(json: generation.text))
  }
}
