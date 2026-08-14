import Foundation

// MARK: - DetectedModel

public enum DetectedModel: String, Hashable, Sendable, CaseIterable {
  case needle2
  case qwen3
  case qwen3P5
  case qwen3P5VL
  case lfm2
  case functionGemma
  case gemma4
  case granite
  case graniteMoeHybrid
  case miniCPM5
  case lfm2P5VL
  case genericLLM
  case genericVLM

  public enum Modality: String, Hashable, Sendable, CaseIterable {
    case text
    case vision
  }

  public var displayName: String {
    switch self {
    case .needle2: "Needle 2"
    case .qwen3: "Qwen3"
    case .qwen3P5: "Qwen3.5"
    case .qwen3P5VL: "Qwen3.5 VL"
    case .lfm2: "LFM2"
    case .functionGemma: "FunctionGemma"
    case .gemma4: "Gemma4"
    case .granite: "Granite"
    case .graniteMoeHybrid: "Granite MoE Hybrid"
    case .miniCPM5: "MiniCPM5"
    case .lfm2P5VL: "LFM2.5 VL"
    case .genericLLM: "Generic LLM"
    case .genericVLM: "Generic VLM"
    }
  }

  public var modality: Modality {
    switch self {
    case .needle2, .qwen3, .qwen3P5, .lfm2, .functionGemma, .granite, .graniteMoeHybrid,
      .miniCPM5, .genericLLM:
      .text
    case .qwen3P5VL, .gemma4, .lfm2P5VL, .genericVLM:
      .vision
    }
  }

  public var supportedEngines: [EngineKind] {
    EngineRunner.registeredEngines(for: self)
  }
}

// MARK: - EngineKind

public enum EngineKind: String, CaseIterable, Sendable {
  case mlx
  case needle2

  public init?(argument: String) {
    switch argument.lowercased().filter({ $0.isLetter || $0.isNumber }) {
    case "mlx": self = .mlx
    case "needle2": self = .needle2
    default: return nil
    }
  }

  public var isAvailable: Bool {
    switch self {
    case .mlx:
      #if canImport(MLX)
        true
      #else
        false
      #endif
    case .needle2:
      if #available(macOS 26, *) {
        true
      } else {
        false
      }
    }
  }

  fileprivate func hasWeights(in files: [String]) -> Bool {
    switch self {
    case .mlx: files.contains { $0.hasSuffix(".safetensors") }
    case .needle2: true
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
    let available = self.engines.filter(self.model.supportedEngines.contains)
    return [.needle2, .mlx].first(where: available.contains)
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
  let nameOrPath: String?

  enum CodingKeys: String, CodingKey {
    case modelType = "model_type"
    case nameOrPath = "name_or_path"
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

  switch (header.modelType, header.nameOrPath) {
  case ("needle", let name?) where name.lowercased().contains("needle2"):
    return .needle2
  case ("qwen3", _): return .qwen3
  case ("qwen3_5", _), ("qwen3_5_text", _):
    return hasProcessorConfiguration(files) ? .qwen3P5VL : .qwen3P5
  case ("lfm2", _): return .lfm2
  case ("lfm2_vl", _), ("lfm2-vl", _): return .lfm2P5VL
  case ("gemma3", _), ("gemma3_text", _): return .functionGemma
  case ("gemma4", _), ("gemma4_unified", _): return .gemma4
  case ("granite", _): return .granite
  case ("granitemoehybrid", _): return .graniteMoeHybrid
  case ("llama", _) where chatTemplate(in: directory, files: files)?.contains("<function") == true:
    return .miniCPM5
  default:
    return hasProcessorConfiguration(files) ? .genericVLM : .genericLLM
  }
}

private func hasProcessorConfiguration(_ files: [String]) -> Bool {
  files.contains { $0 == "preprocessor_config.json" || $0 == "processor_config.json" }
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
