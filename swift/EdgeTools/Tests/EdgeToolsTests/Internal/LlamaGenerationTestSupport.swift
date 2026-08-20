#if HuggingFaceTokenizers && Llama && XGrammar && canImport(CLlama)
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
      EdgeToolsTranscriptContextParameters(transcript: turn.transcript),
      tools: [DefinitionTool(.weatherTest)]
    )
    return try await completeToolTurn(
      in: context,
      tool: .weatherTest,
      toolResponse: .weatherTestResponse,
      generatingToolCall: {
        try engine.generate(
          prompt: .user(turn.userMessage),
          parameters: DefaultLlamaGenerateParameters(
            sampling: .greedy,
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: 256
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: { toolMessage in
        try engine.generate(
          prompt: .tools([toolMessage]),
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
      EdgeToolsTranscriptContextParameters(transcript: turn.transcript),
      tools: [DefinitionTool(.llamaColorTest)]
    )
    let result = try await completeToolTurn(
      in: context,
      tool: .llamaColorTest,
      toolResponse: ["color": "red"],
      generatingToolCall: {
        try engine.generate(
          prompt: .user(turn.userMessage),
          parameters: DefaultLlamaGenerateParameters(
            sampling: .greedy,
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: 128
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: { toolMessage in
        try engine.generate(
          prompt: .tools([toolMessage]),
          parameters: DefaultLlamaGenerateParameters(sampling: .greedy, maxTokens: 64),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      }
    )
    return LlamaVLMToolTurnSnapshot(toolCalls: result.toolCalls, response: result.response)
  }

  func completeAudioToneTurn<Profile: LlamaModelProfile>(
    using engine: LlamaEngine<Profile>
  ) async throws -> LlamaVLMToolTurnSnapshot
  where Profile.GenerateParameters == DefaultLlamaGenerateParameters {
    let turn = try splitUserMessage(
      from: EdgeToolsTranscript(messages: [
        .system(
          "Listen to the audio and call reportAudio with whether it contains silence or a steady tone. After the tool result, summarize it."
        ),
        .user("Classify the audio.", audio: [llamaToneAudioAsset()])
      ])
    )
    let context = engine.context(
      EdgeToolsTranscriptContextParameters(transcript: turn.transcript),
      tools: [DefinitionTool(.llamaAudioTest)]
    )
    let result = try await completeToolTurn(
      in: context,
      tool: .llamaAudioTest,
      toolResponse: ["kind": "tone"],
      generatingToolCall: {
        try engine.generate(
          prompt: .user(turn.userMessage),
          parameters: DefaultLlamaGenerateParameters(
            sampling: .greedy,
            constraint: .toolsWithGrammar(range: .exact(1)),
            maxTokens: 128
          ),
          context: context,
          channel: EdgeToolsGenerationChannel()
        )
      },
      generatingResponse: { toolMessage in
        try engine.generate(
          prompt: .tools([toolMessage]),
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

    fileprivate static let llamaAudioTest = Self(
      name: "reportAudio",
      description: "Reports whether audio contains silence or a steady tone.",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties([
          "kind": EdgeToolsGenerationSchema(.string, .enum(["silence", "tone"]))
        ]),
        .required(["kind"]),
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

  func llamaToneAudioAsset() -> EdgeToolsTranscript.Asset {
    let sampleRate = 16_000
    let samples = (0..<(sampleRate * 2))
      .map { index in
        let phase = 2 * Double.pi * 440 * Double(index) / Double(sampleRate)
        return Int16((sin(phase) * 12_000).rounded())
      }
    let dataSize = samples.count * MemoryLayout<Int16>.size
    var bytes = [UInt8]()
    bytes.reserveCapacity(44 + dataSize)
    bytes.append(contentsOf: "RIFF".utf8)
    appendLittleEndian(UInt32(36 + dataSize), to: &bytes)
    bytes.append(contentsOf: "WAVEfmt ".utf8)
    appendLittleEndian(UInt32(16), to: &bytes)
    appendLittleEndian(UInt16(1), to: &bytes)
    appendLittleEndian(UInt16(1), to: &bytes)
    appendLittleEndian(UInt32(sampleRate), to: &bytes)
    appendLittleEndian(UInt32(sampleRate * MemoryLayout<Int16>.size), to: &bytes)
    appendLittleEndian(UInt16(MemoryLayout<Int16>.size), to: &bytes)
    appendLittleEndian(UInt16(16), to: &bytes)
    bytes.append(contentsOf: "data".utf8)
    appendLittleEndian(UInt32(dataSize), to: &bytes)
    for sample in samples {
      appendLittleEndian(sample, to: &bytes)
    }
    return EdgeToolsTranscript.Asset(bytes: bytes, mimeTypeOverride: .wav)
  }

  private func appendLittleEndian<Value: FixedWidthInteger>(
    _ value: Value,
    to bytes: inout [UInt8]
  ) {
    withUnsafeBytes(of: value.littleEndian) { bytes.append(contentsOf: $0) }
  }
#endif
