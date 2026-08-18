#if Llama && XGrammar && canImport(CLlama)
  import CoreGraphics
  import EdgeTools
  import Foundation
  import ImageIO
  import UniformTypeIdentifiers

  func completeWeatherTurn<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> EdgeToolsTranscript
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    let turn = try splitUserMessage(from: .weatherTest)
    let context = engine.context(
      EdgeToolsTranscriptContextParameters(transcript: turn.transcript)
    )
    return try await completeToolTurn(
      in: context,
      tool: .weatherTest,
      toolResponse: .weatherTestResponse,
      generatingToolCall: {
        try engine.generate(
          prompt: turn.userMessage,
          tools: [.weatherTest],
          parameters: DefaultLlamaGenerateParameters(
            sampling: .greedy,
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: 256
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: {
        try engine.generate(
          tools: [],
          parameters: DefaultLlamaGenerateParameters(maxTokens: 64),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      }
    )
    .transcript
  }

  func generateReasoning<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> EdgeToolsEngineGeneration
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    try await reasoningGeneration(
      from: try engine.generate(
        prompt: .user(.reasoningTest),
        tools: [],
        parameters: DefaultLlamaGenerateParameters(maxTokens: 512),
        context: engine.context(
          EdgeToolsTranscriptContextParameters(reasoningEffort: .high)
        ),
        channel: EdgeToolsGenerationChannel()
      )
    )
  }

  struct LlamaVLMToolTurnSnapshot: Hashable, Sendable {
    var toolCalls: [EdgeRawToolCall]
    var response: String
  }

  func describeRedImage<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> String
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    let task = try engine.generate(
      prompt: .user(
        "What is the dominant color in this image? Answer briefly.",
        images: [try llamaRedImageAsset()]
      ),
      tools: [],
      parameters: DefaultLlamaGenerateParameters(sampling: .greedy, maxTokens: 64),
      context: engine.context(),
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value.response
  }

  func completeImageColorTurn<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> LlamaVLMToolTurnSnapshot
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    let turn = try splitUserMessage(
      from: EdgeToolsTranscript(messages: [
        .system(
          "Inspect the image and call reportColor with its dominant color. After the tool result, summarize it."
        ),
        .user("Report the dominant image color.", images: [try llamaRedImageAsset()])
      ])
    )
    let context = engine.context(
      EdgeToolsTranscriptContextParameters(transcript: turn.transcript)
    )
    let result = try await completeToolTurn(
      in: context,
      tool: .llamaColorTest,
      toolResponse: ["color": "red"],
      generatingToolCall: {
        try engine.generate(
          prompt: turn.userMessage,
          tools: [.llamaColorTest],
          parameters: DefaultLlamaGenerateParameters(
            sampling: .greedy,
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: 128
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: {
        try engine.generate(
          tools: [],
          parameters: DefaultLlamaGenerateParameters(sampling: .greedy, maxTokens: 64),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      }
    )
    return LlamaVLMToolTurnSnapshot(toolCalls: result.toolCalls, response: result.response)
  }

  // MARK: - VLM Fixtures

  extension EdgeToolDefinition {
    fileprivate static let llamaColorTest = Self(
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

  private enum LlamaGenerationTestError: Error {
    case failedToCreateImage
  }

  func llamaRedImageAsset() throws -> EdgeToolsTranscript.Asset {
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
      throw LlamaGenerationTestError.failedToCreateImage
    }
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    guard let image = context.makeImage() else {
      throw LlamaGenerationTestError.failedToCreateImage
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
      throw LlamaGenerationTestError.failedToCreateImage
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw LlamaGenerationTestError.failedToCreateImage
    }
    return EdgeToolsTranscript.Asset(bytes: Array(data as Data), mimeTypeOverride: .png)
  }
#endif
