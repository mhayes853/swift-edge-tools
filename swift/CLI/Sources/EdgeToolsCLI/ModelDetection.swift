import Foundation

// MARK: - DetectedModel

public enum DetectedModel: String, Hashable, Sendable, CaseIterable {
  case needle
  case qwen3
  case qwen3P5
  case lfm2
  case functionGemma
  case granite
  case graniteMoeHybrid
  case miniCPM5

  public var displayName: String {
    switch self {
    case .needle: "Needle"
    case .qwen3: "Qwen3"
    case .qwen3P5: "Qwen3.5"
    case .lfm2: "LFM2"
    case .functionGemma: "FunctionGemma"
    case .granite: "Granite"
    case .graniteMoeHybrid: "Granite MoE Hybrid"
    case .miniCPM5: "MiniCPM5"
    }
  }

  public var supportedEngines: [EngineKind] {
    switch self {
    case .needle: [.mlx, .onnx, .coreml, .coreai]
    default: [.mlx]
    }
  }
}

// MARK: - EngineKind

public enum EngineKind: String, CaseIterable, Sendable {
  case mlx
  case onnx
  case coreml
  case coreai

  public init?(argument: String) {
    switch argument.lowercased().filter({ $0.isLetter || $0.isNumber }) {
    case "mlx": self = .mlx
    case "onnx": self = .onnx
    case "coreml": self = .coreml
    case "coreai": self = .coreai
    default: return nil
    }
  }

  public var isExperimental: Bool {
    self == .coreai
  }

  fileprivate func hasWeights(in files: [String]) -> Bool {
    switch self {
    case .mlx: files.contains { $0.hasSuffix(".safetensors") }
    case .onnx: files.contains { $0.hasSuffix(".onnx") }
    case .coreml: files.contains { $0.hasSuffix(".mlmodelc") || $0.hasSuffix(".mlpackage") }
    case .coreai: files.contains { $0.hasSuffix(".aimodel") || $0.hasSuffix(".aimodelc") }
    }
  }
}

// MARK: - ModelDetection

public struct ModelDetection: Hashable, Sendable {
  public var directory: URL
  public var model: DetectedModel
  public var engines: [EngineKind]
  public var files: [String]

  public init(directory: URL, model: DetectedModel, engines: [EngineKind], files: [String]) {
    self.directory = directory
    self.model = model
    self.engines = engines
    self.files = files
  }

  public var defaultEngine: EngineKind? {
    self.engines.first { !$0.isExperimental }
  }

  public var unavailableEngines: [EngineKind] {
    self.model.supportedEngines.filter { !self.engines.contains($0) }
  }
}

// MARK: - Detecting

extension ModelDetection {
  public static func detect(in directory: URL) throws -> Self {
    let files = (try? FileManager.default.contentsOfDirectory(atPath: directory.path()))?.sorted()
    guard let files, !files.isEmpty else {
      throw EdgeCLIError("Model directory \(directory.path()) is empty.")
    }
    let model = try detectedModel(in: directory, files: files)
    return Self(
      directory: directory,
      model: model,
      engines: model.supportedEngines.filter { $0.hasWeights(in: files) },
      files: files
    )
  }
}

// MARK: - ConfigurationHeader

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

private func detectedModel(in directory: URL, files: [String]) throws -> DetectedModel {
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
  case "qwen3_5", "qwen3_5_text": return .qwen3P5
  case "lfm2": return .lfm2
  case "gemma3", "gemma3_text": return .functionGemma
  case "granite": return .granite
  case "granitemoehybrid": return .graniteMoeHybrid
  case "llama" where chatTemplate(in: directory, files: files)?.contains("<function") == true:
    return .miniCPM5
  default:
    let described = header.modelType ?? header.architectures?.first ?? "unspecified"
    throw EdgeCLIError(
      """
      Unsupported model \(described) in \(directory.path()). Supported models: \
      \(DetectedModel.allCases.map(\.displayName).joined(separator: ", ")).
      """
    )
  }
}

private func chatTemplate(in directory: URL, files: [String]) -> String? {
  if files.contains("chat_template.jinja"),
    let template = try? String(
      contentsOf: directory.appending(path: "chat_template.jinja"),
      encoding: .utf8
    )
  {
    return template
  }
  guard files.contains("tokenizer_config.json"),
    let data = try? Data(contentsOf: directory.appending(path: "tokenizer_config.json")),
    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
  else {
    return nil
  }
  return json["chat_template"] as? String
}
