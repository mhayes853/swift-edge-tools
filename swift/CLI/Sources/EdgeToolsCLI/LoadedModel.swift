import EdgeTools
import Foundation

// MARK: - LoadedModel

struct LoadedModel {
  var detection: ModelDetection
  var engine: EngineKind
  var runner: EngineRunner
  var loadDuration: Duration
}

// MARK: - Loading

extension LoadedModel {
  static func load(
    context: EdgeContext,
    source: ModelSource,
    requestedEngine: EngineKind?,
    hardwareUnit: MLXHardwareUnit,
    quiet: Bool
  ) async throws -> Self {
    let clock = ContinuousClock()
    let start = clock.now
    let directory = try await context.resolveDirectory(source) { repo in
      if !quiet { warn("downloading \(repo)...") }
    }
    let detection = try context.detectModel(directory)
    let engine = try resolvedEngine(requested: requestedEngine, detection: detection)
    if engine.isExperimental, !quiet {
      warn("the \(engine.rawValue) engine is experimental.")
    }
    return Self(
      detection: detection,
      engine: engine,
      runner: try await context.makeRunner(detection, engine, hardwareUnit),
      loadDuration: start.duration(to: clock.now)
    )
  }
}

// MARK: - Requests

extension LoadedModel {
  func makeRequest(
    settings: GenerationSettings,
    prompt: String,
    tools: [EdgeToolDefinition]
  ) throws -> GenerationRequest {
    guard settings.temperature == 0 || self.runner.supportsSampling else {
      throw EdgeCLIError(
        """
        \(self.detection.model.displayName) on \(self.engine.rawValue) always samples greedily; \
        --temperature and --top-p do not apply.
        """
      )
    }
    guard settings.grammar == .auto || self.runner.supportsCustomGrammar else {
      throw EdgeCLIError(
        """
        \(self.detection.model.displayName) on \(self.engine.rawValue) only supports \
        `--grammar auto`; its generate parameters expose a tool call range rather than a full \
        generation constraint.
        """
      )
    }
    return GenerationRequest(
      system: settings.system,
      user: prompt,
      tools: tools,
      grammar: settings.grammar,
      toolCallRange: settings.toolCallRange,
      maxTokens: settings.maxTokens,
      temperature: settings.temperature,
      topP: settings.topP
    )
  }
}

private func resolvedEngine(
  requested: EngineKind?,
  detection: ModelDetection
) throws -> EngineKind {
  if let requested {
    guard detection.engines.contains(requested) else {
      throw EdgeCLIError(
        """
        The \(requested.rawValue) engine has no weights for \
        \(detection.model.displayName) here. Available: \
        \(detection.engines.map(\.rawValue).joined(separator: ", ")).
        """
      )
    }
    return requested
  }
  if let defaultEngine = detection.defaultEngine {
    return defaultEngine
  }
  let experimental = detection.engines.filter(\.isExperimental).map(\.rawValue)
  throw EdgeCLIError(
    """
    No usable engine for \(detection.model.displayName) in \(detection.directory.path()). \
    \(experimental.isEmpty
      ? "Supported engines: \(detection.model.supportedEngines.map(\.rawValue).joined(separator: ", "))."
      : "Select one explicitly with --engine: \(experimental.joined(separator: ", ")).")
    """
  )
}

func warn(_ message: String) {
  FileHandle.standardError.write(Data("warning: \(message)\n".utf8))
}
