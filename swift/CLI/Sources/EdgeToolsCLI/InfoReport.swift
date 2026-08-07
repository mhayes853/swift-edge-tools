import Foundation

// MARK: - InfoReport

public struct InfoReport: Encodable {
  public let directory: String
  public let model: String
  public let engines: [String]
  public let unavailableEngines: [String]
  public let defaultEngine: String?
  public let experimentalEngines: [String]
  public let files: [String]

  public init(detection: ModelDetection) {
    self.directory = detection.directory.path()
    self.model = detection.model.displayName
    self.engines = detection.engines.map(\.rawValue)
    self.unavailableEngines = detection.unavailableEngines.map(\.rawValue)
    self.defaultEngine = detection.defaultEngine?.rawValue
    self.experimentalEngines = detection.engines.filter(\.isExperimental).map(\.rawValue)
    self.files = detection.files
  }
}

// MARK: - Rendering

extension InfoReport {
  public func displayText() -> String {
    let described = self.engines.map {
      self.experimentalEngines.contains($0) ? "\($0) (experimental)" : $0
    }
    var lines = [
      self.directory,
      "  model       \(self.model)",
      "  engines     \(described.isEmpty ? "none" : described.joined(separator: " · "))"
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
