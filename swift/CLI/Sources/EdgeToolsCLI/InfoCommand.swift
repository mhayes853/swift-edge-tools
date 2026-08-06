import ArgumentParser
import Foundation

// MARK: - InfoCommand

struct InfoCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "info",
    abstract: "Report what model and engines were detected, without running anything."
  )

  @OptionGroup var model: ModelOptions

  func run() async throws {
    let directory = try await self.model.source.resolve(
      onDownloadStart: { output("Downloading \($0)...") }
    )
    let detection = try ModelDetection.detect(in: directory)
    let unavailable = detection.model.supportedEngines.filter { !detection.engines.contains($0) }

    output(directory.path())
    output("  model       \(detection.model.displayName)")
    if detection.model.isGenericFallback {
      output("              no tool call grammar or parser; tool calls will not be parsed")
    }
    let engines = detection.engines.map {
      $0.isExperimental ? "\($0.rawValue) (experimental)" : $0.rawValue
    }
    output("  engines     \(engines.isEmpty ? "none" : engines.joined(separator: " · "))")
    if let defaultEngine = detection.defaultEngine {
      output("              defaults to \(defaultEngine.rawValue)")
    }
    if !unavailable.isEmpty {
      output(
        "              \(unavailable.map(\.rawValue).joined(separator: " · ")) — no weights here"
      )
    }
    output("  files       \(detection.files.joined(separator: " · "))")
  }
}
