import CustomDump
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `ModelDetection tests` {
  @Test
  func `Detects Needle From Encoder Decoder Configuration`() throws {
    let directory = try temporaryModel(
      configuration: """
        {"num_encoder_layers": 4, "num_decoder_layers": 4, "decoder_start_token_id": 1}
        """,
      files: ["model.safetensors"]
    )
    let detection = try ModelDetection.detect(in: directory)

    expectNoDifference(detection.model, .needle)
    expectNoDifference(detection.engines, [.mlx])
  }

  @Test
  func `Detects Known Model Types`() throws {
    let expected: [String: DetectedModel] = [
      "qwen3": .qwen3,
      "qwen3_5": .qwen3P5,
      "lfm2": .lfm2,
      "gemma3_text": .functionGemma,
      "granite": .granite,
      "granitemoehybrid": .graniteMoeHybrid
    ]
    for (modelType, model) in expected {
      let directory = try temporaryModel(
        configuration: "{\"model_type\": \"\(modelType)\"}",
        files: ["model.safetensors"]
      )
      expectNoDifference(try ModelDetection.detect(in: directory).model, model)
    }
  }

  @Test
  func `Detects MiniCPM5 By Its Chat Template Markers`() throws {
    let directory = try temporaryModel(
      configuration: "{\"model_type\": \"llama\"}",
      files: ["model.safetensors"],
      chatTemplate: "{% for message in messages %}<function=get_weather>{% endfor %}"
    )

    expectNoDifference(try ModelDetection.detect(in: directory).model, .miniCPM5)
  }

  @Test
  func `Rejects A Plain Llama Without MiniCPM5 Markers`() throws {
    let directory = try temporaryModel(
      configuration: "{\"model_type\": \"llama\"}",
      files: ["model.safetensors"],
      chatTemplate: "{% for message in messages %}{{ message.content }}{% endfor %}"
    )

    #expect(throws: EdgeCLIError.self) {
      try ModelDetection.detect(in: directory)
    }
  }

  @Test
  func `Throws For Unsupported Architectures`() throws {
    let directory = try temporaryModel(
      configuration: "{\"model_type\": \"llama\"}",
      files: ["model.safetensors"]
    )

    #expect(throws: EdgeCLIError.self) {
      try ModelDetection.detect(in: directory)
    }
  }

  @Test
  func `Only Reports Engines That Have Weights Present`() throws {
    let directory = try temporaryModel(
      configuration: """
        {"num_encoder_layers": 4, "num_decoder_layers": 4, "decoder_start_token_id": 1}
        """,
      files: ["encoder.onnx", "decoder.onnx"]
    )
    let detection = try ModelDetection.detect(in: directory)

    expectNoDifference(detection.engines, [.onnx])
    expectNoDifference(detection.defaultEngine, .onnx)
  }

  @Test
  func `Detects CoreAI Weights But Never Defaults To Them`() throws {
    let directory = try temporaryModel(
      configuration: """
        {"num_encoder_layers": 4, "num_decoder_layers": 4, "decoder_start_token_id": 1}
        """,
      files: ["encoder.aimodel", "decoder.aimodel"]
    )
    let detection = try ModelDetection.detect(in: directory)

    expectNoDifference(detection.engines, [.coreai])
    expectNoDifference(detection.defaultEngine, nil)
  }

  @Test
  func `Prefers A Stable Engine Over An Experimental One`() throws {
    let directory = try temporaryModel(
      configuration: """
        {"num_encoder_layers": 4, "num_decoder_layers": 4, "decoder_start_token_id": 1}
        """,
      files: ["model.safetensors", "encoder.aimodelc", "decoder.aimodelc"]
    )
    let detection = try ModelDetection.detect(in: directory)

    expectNoDifference(detection.engines, [.mlx, .coreai])
    expectNoDifference(detection.defaultEngine, .mlx)
  }

  @Test
  func `Throws When There Is No Configuration File`() throws {
    let directory = try temporaryModel(configuration: nil, files: ["model.safetensors"])

    #expect(throws: EdgeCLIError.self) {
      try ModelDetection.detect(in: directory)
    }
  }
}

private func temporaryModel(
  configuration: String?,
  files: [String],
  chatTemplate: String? = nil
) throws -> URL {
  let directory = URL.temporaryDirectory.appending(path: "edge-tests-\(UUID().uuidString)")
  try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
  if let configuration {
    try Data(configuration.utf8).write(to: directory.appending(path: "config.json"))
  }
  if let chatTemplate {
    try Data(chatTemplate.utf8).write(to: directory.appending(path: "chat_template.jinja"))
  }
  for file in files {
    try Data().write(to: directory.appending(path: file))
  }
  return directory
}
