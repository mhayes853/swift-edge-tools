#if MLX && XGrammar && canImport(MLX)
  import CoreGraphics
  import EdgeTools
  import Foundation
  import ImageIO
  import UniformTypeIdentifiers

  private enum MLXGenerationTestError: Error {
    case missingToolCall
    case missingFinalResponse
    case failedToCreateImage
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

    fileprivate static let colorTest = Self(
      name: "reportColor",
      description: "Reports the dominant color visible in an image.",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties([
          "color": EdgeToolsGenerationSchema(
            .string,
            .enum(["red", "green", "blue", "yellow", "orange", "purple", "black", "white", "gray"])
          )
        ]),
        .required(["color"]),
        .additionalProperties(false)
      )
    )
  }

  struct VLMToolTurnSnapshot: Hashable, Sendable {
    var toolCalls: [EdgeRawToolCall]
    var response: String
  }

  func describeRedImage<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> String
  where
    Engine.Prompt == EdgeToolsLLMPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let prompt = EdgeToolsLLMPrompt(messages: [
      .user(
        "What is the dominant color in this image? Answer briefly.",
        images: [try redImageAsset()]
      )
    ])
    let task = try engine.generate(
      prompt: prompt,
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 64),
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value.response
  }

  func completeImageColorTurn<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> VLMToolTurnSnapshot
  where
    Engine.Prompt == EdgeToolsLLMPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    var prompt = EdgeToolsLLMPrompt(messages: [
      .system(
        "Inspect the image and call reportColor with its dominant color. After the tool result, summarize it."
      ),
      .user("Report the dominant image color.", images: [try redImageAsset()])
    ])
    let toolTask = try engine.generate(
      prompt: prompt,
      tools: [.colorTest],
      parameters: DefaultMLXGenerateParameters(
        constraint: .toolsWithGrammar(range: .exact(1)),
        maxTokens: 128
      ),
      channel: EdgeToolsGenerationChannel()
    )
    let toolGeneration = try await toolTask.value
    guard !toolGeneration.toolCalls.isEmpty else {
      throw MLXGenerationTestError.missingToolCall
    }

    prompt.messages.append(.assistant(toolCalls: toolGeneration.toolCalls))
    prompt.messages.append(.tool(name: "reportColor", response: ["color": "red"]))
    let responseTask = try engine.generate(
      prompt: prompt,
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 64),
      channel: EdgeToolsGenerationChannel()
    )
    let response = try await responseTask.value.response
    guard !response.isEmpty else { throw MLXGenerationTestError.missingFinalResponse }
    return VLMToolTurnSnapshot(toolCalls: toolGeneration.toolCalls, response: response)
  }

  private func redImageAsset() throws -> EdgeToolsLLMPrompt.Asset {
    let width = 128
    let height = 128
    guard
      let context = CGContext(
        data: nil,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: width * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else {
      throw MLXGenerationTestError.failedToCreateImage
    }
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    guard let image = context.makeImage() else {
      throw MLXGenerationTestError.failedToCreateImage
    }

    let data = NSMutableData()
    guard
      let destination = CGImageDestinationCreateWithData(
        data,
        UTType.png.identifier as CFString,
        1,
        nil
      )
    else {
      throw MLXGenerationTestError.failedToCreateImage
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw MLXGenerationTestError.failedToCreateImage
    }
    return EdgeToolsLLMPrompt.Asset(bytes: Array(data as Data), mimeTypeOverride: .png)
  }
#endif
