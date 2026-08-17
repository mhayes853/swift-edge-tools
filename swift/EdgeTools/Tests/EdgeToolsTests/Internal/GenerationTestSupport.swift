#if XGrammar && !os(WASI)
  import EdgeTools

  enum GenerationTestError: Error {
    case missingToolCall
    case missingFinalResponse
    case missingReasoning
    case missingUserMessage
  }

  struct ToolTurnSnapshot {
    var transcript: EdgeToolsTranscript
    var toolCalls: [EdgeRawToolCall]
    var response: String
  }

  // MARK: - Shared Turns

  func completeToolTurn<ModelState>(
    in context: EdgeToolsTranscriptContext<ModelState>,
    tool: EdgeToolDefinition,
    toolResponse: EdgeToolsValue,
    generatingToolCall: () throws -> AnyGenerationTask,
    generatingResponse: () throws -> AnyGenerationTask
  ) async throws -> ToolTurnSnapshot {
    let toolGeneration = try await generatingToolCall().value
    guard !toolGeneration.toolCalls.isEmpty else {
      throw GenerationTestError.missingToolCall
    }

    context.transcript.messages.append(.tool(name: tool.name, response: toolResponse))

    let responseGeneration = try await generatingResponse().value
    guard !responseGeneration.response.isEmpty else {
      throw GenerationTestError.missingFinalResponse
    }
    return ToolTurnSnapshot(
      transcript: context.transcript,
      toolCalls: toolGeneration.toolCalls,
      response: responseGeneration.response
    )
  }

  func reasoningGeneration(from task: AnyGenerationTask) async throws -> EdgeToolsEngineGeneration {
    let generation = try await task.value
    guard !generation.reasoning.isEmpty else {
      throw GenerationTestError.missingReasoning
    }
    return generation
  }

  func splitUserMessage(
    from transcript: EdgeToolsTranscript
  ) throws -> (
    transcript: EdgeToolsTranscript,
    userMessage: EdgeToolsTranscript.UserMessage
  ) {
    var transcript = transcript
    guard case .user(let userMessage) = transcript.messages.popLast() else {
      throw GenerationTestError.missingUserMessage
    }
    return (transcript, userMessage)
  }

  // MARK: - Fixtures

  extension String {
    static let reasoningTest =
      "Think carefully before answering: what is the sum of 19 and 23? Keep the final answer brief."
  }

  extension EdgeToolsTranscript {
    static let weatherTest = Self(messages: [
      .system(
        "Use getWeather when needed. After receiving its result, summarize it for the user."
      ),
      .user("What is the weather in Paris?")
    ])
  }

  extension EdgeToolDefinition {
    static let weatherTest = Self(
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

  extension EdgeToolsValue {
    static let weatherTestResponse: Self = [
      "location": "Paris",
      "condition": "sunny",
      "temperatureCelsius": 21
    ]
  }
#endif
