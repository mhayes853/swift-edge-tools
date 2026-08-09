#if MLX && XGrammar && canImport(MLX)
  import CoreGraphics
  import AVFoundation
  import EdgeTools
  import Foundation
  import ImageIO
  import UniformTypeIdentifiers

  private enum MLXGenerationTestError: Error {
    case missingToolCall
    case missingFinalResponse
    case missingReasoning
    case failedToCreateImage
    case failedToCreateVideo
    case failedToAppendVideoFrame
    case failedToFinishVideo
  }

  func completeWeatherTurn<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> EdgeToolsConversationalPrompt
  where
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    try await completeToolTurn(
      using: engine,
      prompt: .weatherTest,
      tool: .weatherTest,
      toolResponse: [
        "location": "Paris",
        "condition": "sunny",
        "temperatureCelsius": 21
      ],
      toolMaxTokens: 256
    )
    .transcript
  }

  func generateReasoning<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> EdgeToolsEngineGeneration
  where
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let prompt = EdgeToolsConversationalPrompt(
      messages: [
        .user(
          "Think carefully before answering: what is the sum of 19 and 23? Keep the final answer brief."
        )
      ],
      reasoningEffort: .high
    )
    let task = try engine.generate(
      prompt: prompt,
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 512),
      channel: EdgeToolsGenerationChannel()
    )
    let generation = try await task.value
    guard !generation.reasoning.isEmpty else {
      throw MLXGenerationTestError.missingReasoning
    }
    return generation
  }

  extension EdgeToolsConversationalPrompt {
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
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let prompt = EdgeToolsConversationalPrompt(messages: [
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
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let turn = try await completeToolTurn(
      using: engine,
      prompt: EdgeToolsConversationalPrompt(messages: [
        .system(
          "Inspect the image and call reportColor with its dominant color. After the tool result, summarize it."
        ),
        .user("Report the dominant image color.", images: [try redImageAsset()])
      ]),
      tool: .colorTest,
      toolResponse: ["color": "red"],
      toolMaxTokens: 128
    )
    return VLMToolTurnSnapshot(toolCalls: turn.toolCalls, response: turn.response)
  }

  private func completeToolTurn<Engine: EdgeToolsEngine>(
    using engine: Engine,
    prompt: EdgeToolsConversationalPrompt,
    tool: EdgeToolDefinition,
    toolResponse: EdgeToolsValue,
    toolMaxTokens: Int
  ) async throws -> (
    transcript: EdgeToolsConversationalPrompt,
    toolCalls: [EdgeRawToolCall],
    response: String
  )
  where
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    var transcript = prompt
    let toolGenerationTask = try engine.generate(
      prompt: transcript,
      tools: [tool],
      parameters: DefaultMLXGenerateParameters(
        constraint: .toolsWithGrammar(range: .exact(1)),
        maxTokens: toolMaxTokens
      ),
      channel: EdgeToolsGenerationChannel()
    )
    let toolGeneration = try await toolGenerationTask.value
    guard !toolGeneration.toolCalls.isEmpty else {
      throw MLXGenerationTestError.missingToolCall
    }

    transcript.messages.append(.init(generation: toolGeneration))
    transcript.messages.append(.tool(name: tool.name, response: toolResponse))

    let responseGenerationTask = try engine.generate(
      prompt: transcript,
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 64),
      channel: EdgeToolsGenerationChannel()
    )
    let responseGeneration = try await responseGenerationTask.value
    guard !responseGeneration.response.isEmpty else {
      throw MLXGenerationTestError.missingFinalResponse
    }

    transcript.messages.append(.init(generation: responseGeneration))
    return (
      transcript: transcript,
      toolCalls: toolGeneration.toolCalls,
      response: responseGeneration.response
    )
  }

  func describeRedVideo<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> String
  where
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let video = try await redVideoAsset()
    defer { video.remove() }
    let prompt = EdgeToolsConversationalPrompt(messages: [
      .user(
        "What is the dominant color in this video? Answer briefly.",
        videos: [video.asset]
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

  func describeRedImageAndVideo<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> String
  where
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let video = try await redVideoAsset()
    defer { video.remove() }
    let prompt = EdgeToolsConversationalPrompt(messages: [
      .user(
        "What is the dominant color across the image and video? Answer briefly.",
        images: [try redImageAsset()],
        videos: [video.asset]
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

  func completeVideoColorTurn<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> VLMToolTurnSnapshot
  where
    Engine.Prompt == EdgeToolsConversationalPrompt,
    Engine.GenerateParameters == DefaultMLXGenerateParameters
  {
    let video = try await redVideoAsset()
    defer { video.remove() }
    let turn = try await completeToolTurn(
      using: engine,
      prompt: EdgeToolsConversationalPrompt(messages: [
        .system(
          "Inspect the video and call reportColor with its dominant color. After the tool result, summarize it."
        ),
        .user("Report the dominant video color.", videos: [video.asset])
      ]),
      tool: .colorTest,
      toolResponse: ["color": "red"],
      toolMaxTokens: 128
    )
    return VLMToolTurnSnapshot(toolCalls: turn.toolCalls, response: turn.response)
  }

  private func redImageAsset() throws -> EdgeToolsConversationalPrompt.Asset {
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
    return EdgeToolsConversationalPrompt.Asset(bytes: Array(data as Data), mimeTypeOverride: .png)
  }

  private struct VideoTestAsset {
    let asset: EdgeToolsConversationalPrompt.Asset
    let url: URL

    func remove() {
      try? FileManager.default.removeItem(at: self.url)
    }
  }

  private func redVideoAsset() async throws -> VideoTestAsset {
    let width = 128
    let height = 128
    let url = FileManager.default.temporaryDirectory
      .appending(path: "EdgeToolsTests-\(UUID().uuidString).mp4")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width,
        AVVideoHeightKey: height
      ]
    )
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
      assetWriterInput: input,
      sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height
      ]
    )
    guard writer.canAdd(input) else { throw MLXGenerationTestError.failedToCreateVideo }
    writer.add(input)
    guard writer.startWriting() else { throw MLXGenerationTestError.failedToCreateVideo }
    writer.startSession(atSourceTime: .zero)

    for index in 0..<2 {
      guard let pool = adaptor.pixelBufferPool else {
        throw MLXGenerationTestError.failedToCreateVideo
      }
      var pixelBuffer: CVPixelBuffer?
      guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer) == kCVReturnSuccess,
        let pixelBuffer
      else { throw MLXGenerationTestError.failedToCreateVideo }
      try fill(
        pixelBuffer: pixelBuffer,
        with: CGColor(red: 1, green: 0, blue: 0, alpha: 1),
        width: width,
        height: height
      )
      guard
        adaptor.append(pixelBuffer, withPresentationTime: CMTime(value: Int64(index), timescale: 2))
      else { throw MLXGenerationTestError.failedToAppendVideoFrame }
    }

    input.markAsFinished()
    await withCheckedContinuation { continuation in
      writer.finishWriting {
        continuation.resume()
      }
    }
    guard writer.status == .completed else {
      throw MLXGenerationTestError.failedToFinishVideo
    }
    return VideoTestAsset(
      asset: EdgeToolsConversationalPrompt.Asset(path: url.path(), mimeTypeOverride: .mp4),
      url: url
    )
  }

  private func fill(
    pixelBuffer: CVPixelBuffer,
    with color: CGColor,
    width: Int,
    height: Int
  ) throws {
    guard CVPixelBufferLockBaseAddress(pixelBuffer, []) == kCVReturnSuccess,
      let data = CVPixelBufferGetBaseAddress(pixelBuffer)
    else { throw MLXGenerationTestError.failedToCreateVideo }
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
    guard
      let context = CGContext(
        data: data,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue
      )
    else { throw MLXGenerationTestError.failedToCreateVideo }
    context.setFillColor(color)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
  }
#endif
