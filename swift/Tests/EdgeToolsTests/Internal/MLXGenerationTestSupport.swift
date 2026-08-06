#if MLX && XGrammar && canImport(MLX)
  import EdgeTools

  private enum MLXGenerationTestError: Error {
    case missingToolCall
    case missingFinalResponse
  }

  func completeWeatherTurn<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> EdgeToolsLLMPrompt
  where
    Engine.Prompt == EdgeToolsLLMPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    var transcript = EdgeToolsLLMPrompt.weatherTest
    let toolParameters = DefaultMLXGenerateParameters(
      constraint: .toolsWithGrammar(range: .exact(1)),
      maxTokens: 256
    )
    let responseParameters = DefaultMLXGenerateParameters(
      constraint: .unconstrained,
      maxTokens: 64
    )

    let toolGenerationTask = try engine.generate(
      prompt: transcript,
      tools: [.weatherTest],
      parameters: toolParameters,
      channel: EdgeToolsGenerationChannel()
    )
    let toolGeneration = try await toolGenerationTask.value
    guard !toolGeneration.toolCalls.isEmpty else {
      throw MLXGenerationTestError.missingToolCall
    }
    transcript.messages.append(.assistant(toolCalls: toolGeneration.toolCalls))
    transcript.messages.append(
      .tool(
        name: "getWeather",
        response: [
          "location": "Paris",
          "condition": "sunny",
          "temperatureCelsius": 21
        ]
      )
    )

    let responseGenerationTask = try engine.generate(
      prompt: transcript,
      tools: [],
      parameters: responseParameters,
      channel: EdgeToolsGenerationChannel()
    )
    let responseGeneration = try await responseGenerationTask.value
    let response = responseGeneration.response
    guard !response.isEmpty else { throw MLXGenerationTestError.missingFinalResponse }
    transcript.messages.append(.assistant(response))
    return transcript
  }

  extension EdgeToolsLLMPrompt {
    static let weatherTest = Self(messages: [
      .system(
        "Use getWeather when needed. After receiving its result, summarize it for the user."
      ),
      .user("What is the weather in Paris?")
    ])
  }

  extension EdgeToolDefinition {
    fileprivate static let weatherTest = Self(
      name: "getWeather",
      description: "Gets the weather for a city.",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties([
          "location": EdgeToolsGenerationSchema(
            .string,
            .enum(["Paris"])
          )
        ]),
        .required(["location"]),
        .additionalProperties(false)
      )
    )

  }
#endif
