import Foundation

// MARK: - InfoReport

public struct InfoReport: Encodable {
  public let directory: String
  public let model: String
  public let modality: String
  public let engines: [String]
  public let unavailableEngines: [String]
  public let defaultEngine: String?
  public let files: [String]

  public init(detection: ModelDetection) {
    self.directory = detection.directory.path()
    self.model = detection.model.displayName
    self.modality = detection.model.modality.rawValue
    self.engines = detection.engines.map(\.rawValue)
    self.unavailableEngines = detection.unavailableEngines.map(\.rawValue)
    self.defaultEngine = detection.defaultEngine?.rawValue
    self.files = detection.files
  }
}

// MARK: - Rendering

extension InfoReport {
  public func displayText() -> String {
    var lines = [
      self.directory,
      "  model       \(self.model)",
      "  modality    \(self.modality)",
      "  engines     \(self.engines.isEmpty ? "none" : self.engines.joined(separator: " · "))"
    ]
    if let defaultEngine = self.defaultEngine {
      lines.append("              defaults to \(defaultEngine)")
    }
    if !self.unavailableEngines.isEmpty {
      lines.append(
        "              \(self.unavailableEngines.joined(separator: " · ")) — no weights here"
      )
    }
    lines.append("  files       \(self.files.joined(separator: " · "))")
    return lines.joined(separator: "\n")
  }

  public func jsonText() throws -> String {
    try self.encodedJSON()
  }
}
