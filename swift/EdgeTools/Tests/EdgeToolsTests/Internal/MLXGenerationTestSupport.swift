#if MLX && XGrammar && canImport(MLX)
  import CoreGraphics
  import AVFoundation
  import EdgeTools
  import MLXLMCommon
  import Foundation
  import ImageIO
  import UniformTypeIdentifiers

  private enum MLXGenerationTestError: Error {
    case missingToolOutput
    case failedToCreateImage
    case failedToCreateVideo
    case failedToAppendVideoFrame
    case failedToFinishVideo
  }

  func completeWeatherTurn<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> EdgeToolsTranscript
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    try await completeToolTurn(
      using: engine,
      prompt: .weatherTest,
      tool: .weatherTest,
      toolResponse: .weatherTestResponse,
      toolMaxTokens: 256
    )
    .transcript
  }

  func generateReasoning<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> EdgeToolsEngineGeneration
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    try await reasoningGeneration(
      from: try engine.generate(
        prompt: .user(.reasoningTest),
        tools: [],
        parameters: DefaultMLXGenerateParameters(maxTokens: 512),
        context: engine.context(MLXContextParameters(reasoningEffort: .high)),
        channel: EdgeToolsGenerationChannel()
      )
    )
  }

  struct SessionWeatherTurnSnapshot: Hashable, Sendable {
    var toolCalls: [EdgeRawToolCall]
    var toolOutput: String
    var response: String
  }

  func completeWeatherTurn<Profile: MLXModelProfile>(
    using session: EdgeToolsSession<MLXEngine<Profile>>,
    sampling parameters: EdgeToolsFusedSamplingParameters
  ) async throws -> SessionWeatherTurnSnapshot
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let turn = try splitUserMessage(from: .weatherTest)
    let context = session.context(transcript: turn.transcript, reasoningEffort: .none)
    let toolGeneration = try await session.generate(
      prompt: .user(turn.userMessage),
      context: context,
      parameters: DefaultMLXGenerateParameters(
        sampler: MLXFusedSampler(parameters: parameters),
        constraint: .toolsWithGrammar(range: .exact(1)),
        maxTokens: 256
      )
    )
    guard let toolCall = toolGeneration.toolCalls.first else {
      throw GenerationTestError.missingToolCall
    }
    guard let toolOutput = try await toolCall.output as? String else {
      throw MLXGenerationTestError.missingToolOutput
    }

    context.transcript.messages.append(.tool(name: "getWeather", response: .string(toolOutput)))
    let responseTask = try session.engine.generate(
      tools: [],
      parameters: DefaultMLXGenerateParameters(
        sampler: MLXFusedSampler(parameters: parameters),
        constraint: .unconstrained,
        maxTokens: 64
      ),
      context: context,
      channel: EdgeToolsGenerationChannel()
    )
    let response = try await responseTask.value.response
    guard !response.isEmpty else { throw GenerationTestError.missingFinalResponse }
    return SessionWeatherTurnSnapshot(
      toolCalls: toolGeneration.engineGeneration.toolCalls,
      toolOutput: toolOutput,
      response: response
    )
  }

  @EdgeToolsGenerable
  struct WeatherToolArguments: Equatable {
    var location: String
  }

  struct WeatherTestTool: EdgeTool {
    typealias Input = WeatherToolArguments
    typealias Output = String

    let name = "getWeather"
    let description = "Gets the weather for a city."

    func invoke(input: WeatherToolArguments) async throws -> sending String {
      "It is sunny and 21 degrees celsius in \(input.location)."
    }
  }

  extension EdgeToolDefinition {
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

  func describeRedImage<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> String
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let task = try engine.generate(
      prompt: .user(
        "What is the dominant color in this image? Answer briefly.",
        images: [try redImageAsset()]
      ),
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 64),
      context: engine.context(),
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value.response
  }

  func completeImageColorTurn<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> VLMToolTurnSnapshot
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let turn = try await completeToolTurn(
      using: engine,
      prompt: EdgeToolsTranscript(messages: [
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

  private func completeToolTurn<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>,
    prompt: EdgeToolsTranscript,
    tool: EdgeToolDefinition,
    toolResponse: EdgeToolsValue,
    toolMaxTokens: Int
  ) async throws -> ToolTurnSnapshot
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let turn = try splitUserMessage(from: prompt)
    let context = engine.context(MLXContextParameters(transcript: turn.transcript))
    return try await completeToolTurn(
      in: context,
      tool: tool,
      toolResponse: toolResponse,
      generatingToolCall: {
        try engine.generate(
          prompt: .user(turn.userMessage),
          tools: [tool],
          parameters: DefaultMLXGenerateParameters(
            sampler: ArgMaxSampler(),
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: toolMaxTokens
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: {
        try engine.generate(
          tools: [],
          parameters: DefaultMLXGenerateParameters(maxTokens: 64),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      }
    )
  }

  func describeRedVideo<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> String
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let video = try await redVideoAsset()
    defer { video.remove() }
    let prompt = EdgeToolsTranscript.UserMessage(
      "What is the dominant color in this video? Answer briefly.",
      videos: [video.asset]
    )
    let task = try engine.generate(
      prompt: .user(prompt),
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 64),
      context: engine.context(),
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value.response
  }

  func describeRedImageAndVideo<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> String
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let video = try await redVideoAsset()
    defer { video.remove() }
    let prompt = EdgeToolsTranscript.UserMessage(
      "What is the dominant color across the image and video? Answer briefly.",
      images: [try redImageAsset()],
      videos: [video.asset]
    )
    let task = try engine.generate(
      prompt: .user(prompt),
      tools: [],
      parameters: DefaultMLXGenerateParameters(maxTokens: 64),
      context: engine.context(),
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value.response
  }

  func completeVideoColorTurn<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>
  ) async throws -> VLMToolTurnSnapshot
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let video = try await redVideoAsset()
    defer { video.remove() }
    let turn = try await completeToolTurn(
      using: engine,
      prompt: EdgeToolsTranscript(messages: [
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

  private func redImageAsset() throws -> EdgeToolsTranscript.Asset {
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
    return EdgeToolsTranscript.Asset(bytes: Array(data as Data), mimeTypeOverride: .png)
  }

  private struct VideoTestAsset {
    let asset: EdgeToolsTranscript.Asset
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
      asset: EdgeToolsTranscript.Asset(path: url.path(), mimeTypeOverride: .mp4),
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
