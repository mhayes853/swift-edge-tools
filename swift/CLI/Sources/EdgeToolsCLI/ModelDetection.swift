import Foundation

// MARK: - DetectedModel

public enum DetectedModel: Hashable, Sendable {
  case needle
  case qwen3
  case qwen35
  case lfm2
  case functionGemma
  case genericLLM(modelType: String)
}

extension DetectedModel {
  public var displayName: String {
    switch self {
    case .needle: "Needle"
    case .qwen3: "Qwen3"
    case .qwen35: "Qwen3.5"
    case .lfm2: "LFM2"
    case .functionGemma: "FunctionGemma"
    case .genericLLM(let modelType): "unknown (\(modelType)) — generic LLM fallback"
    }
  }

  /// Generic models have no model-specific tool call grammar or parser, so tool calls they emit
  /// cannot be parsed back out of the response.
  public var isGenericFallback: Bool {
    if case .genericLLM = self { return true }
    return false
  }

  public var supportedEngines: [EngineKind] {
    switch self {
    case .needle: [.mlx, .onnx, .coreml, .coreai]
    case .qwen3, .qwen35, .lfm2, .functionGemma, .genericLLM: [.mlx]
    }
  }
}

// MARK: - EngineKind

public enum EngineKind: String, CaseIterable, Sendable {
  case mlx
  case onnx
  case coreml
  case coreai
}

extension EngineKind {
  /// CoreAI support is experimental, so it is never selected without being asked for by name.
  public var isExperimental: Bool {
    self == .coreai
  }
}

// MARK: - ModelDetection

public struct ModelDetection: Sendable {
  public var directory: URL
  public var model: DetectedModel
  public var engines: [EngineKind]
  public var files: [String]

  public var defaultEngine: EngineKind? {
    self.engines.first { !$0.isExperimental }
  }
}

// MARK: - Detecting

extension ModelDetection {
  public static func detect(in directory: URL) throws -> Self {
    let files = (try? FileManager.default.contentsOfDirectory(atPath: directory.path()))?.sorted()
    guard let files, !files.isEmpty else {
      throw EdgeCLIError("Model directory \(directory.path()) is empty.")
    }
    let model = try detectModel(in: directory, files: files)
    let engines = model.supportedEngines.filter { $0.hasWeights(in: files) }
    return Self(directory: directory, model: model, engines: engines, files: files)
  }
}

extension EngineKind {
  func hasWeights(in files: [String]) -> Bool {
    switch self {
    case .mlx: files.contains { $0.hasSuffix(".safetensors") }
    case .onnx: files.contains { $0.hasSuffix(".onnx") }
    case .coreml: files.contains { $0.hasSuffix(".mlmodelc") || $0.hasSuffix(".mlpackage") }
    case .coreai: files.contains { $0.hasSuffix(".aimodel") || $0.hasSuffix(".aimodelc") }
    }
  }
}

// MARK: - Configuration Header

private struct ConfigurationHeader: Decodable {
  let modelType: String?
  let architectures: [String]?
  let encoderLayers: Int?
  let decoderStartTokenId: Int?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case architectures
    case encoderLayers = "num_encoder_layers"
    case decoderStartTokenId = "decoder_start_token_id"
  }
}

private func detectModel(in directory: URL, files: [String]) throws -> DetectedModel {
  let configurationFile = ["configuration.json", "config.json"].first { files.contains($0) }
  guard let configurationFile else {
    throw EdgeCLIError(
      "No config.json or configuration.json in \(directory.path()) — cannot identify the model."
    )
  }
  let data = try Data(contentsOf: directory.appending(path: configurationFile))
  let header = try JSONDecoder().decode(ConfigurationHeader.self, from: data)

  if header.encoderLayers != nil, header.decoderStartTokenId != nil {
    return .needle
  }
  switch header.modelType {
  case "needle": return .needle
  case "qwen3": return .qwen3
  case "qwen3_5", "qwen3_5_text": return .qwen35
  case "lfm2": return .lfm2
  case "gemma3", "gemma3_text": return .functionGemma
  case let modelType?: return .genericLLM(modelType: modelType)
  case nil:
    guard let architecture = header.architectures?.first else {
      throw EdgeCLIError(
        "\(configurationFile) has no model_type or architectures — cannot identify the model."
      )
    }
    return .genericLLM(modelType: architecture)
  }
}
