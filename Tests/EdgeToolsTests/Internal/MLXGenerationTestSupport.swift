#if MLX && XGrammar && canImport(MLX)
  import EdgeTools

  private enum MLXGenerationTestError: Error {
    case missingToolCall
    case missingFinalResponse
  }

  func completeWeatherTurn<Model: EdgeToolsLanguageModel>(
    using engine: EdgeToolsMLXEngine<Model>
  ) async throws -> EdgeToolsLLMPrompt where Model.Prompt == EdgeToolsLLMPrompt {
    let session = EdgeToolsSession(engine: engine)
    var transcript = EdgeToolsLLMPrompt.weatherTest
    let parameters = EdgeToolsMLXEngine<Model>
      .GenerateParameters(
        toolCallRange: .exact(1),
        maxTokens: 256
      )

    let toolGeneration = try await session.generateRawToolCalls(
      prompt: transcript,
      tools: [.weatherTest],
      parameters: parameters
    )
    guard !toolGeneration.toolCalls.isEmpty else {
      throw MLXGenerationTestError.missingToolCall
    }
    transcript.messages.append(.assistant(toolCalls: toolGeneration.toolCalls))
    transcript.messages.append(
      .tool([
        "location": "Paris",
        "condition": "sunny",
        "temperatureCelsius": 21
      ])
    )

    let responseGeneration = try await session.generateRawToolCalls(
      prompt: transcript,
      tools: [.finalResponseTest],
      parameters: parameters
    )
    guard
      let call = responseGeneration.toolCalls.first,
      case .object(let arguments) = call.arguments,
      case .string(let response)? = arguments["response"]
    else {
      throw MLXGenerationTestError.missingFinalResponse
    }
    transcript.messages.append(.assistant(response))
    return transcript
  }

  extension EdgeToolsLLMPrompt {
    static let weatherTest = Self(messages: [
      .system(
        "Use getWeather when needed. After receiving its result, use finalResponse to summarize it."
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

    fileprivate static let finalResponseTest = Self(
      name: "finalResponse",
      description: "Summarizes the tool result for the user.",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties([
          "response": EdgeToolsGenerationSchema(
            .string,
            .enum(["It is sunny and 21°C in Paris."])
          )
        ]),
        .required(["response"]),
        .additionalProperties(false)
      )
    )
  }
#endif
