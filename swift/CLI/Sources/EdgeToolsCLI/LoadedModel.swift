import EdgeTools
import Foundation

// MARK: - LoadedModel

struct LoadedModel {
  var detection: ModelDetection
  var runner: EngineRunner
  var hardwareUnit: MLXHardwareUnit
  var loadDuration: Duration
}

// MARK: - Loading

extension LoadedModel {
  static func load(
    context: EdgeContext,
    source: ModelSource,
    requestedEngine: EngineKind?,
    hardwareUnit: MLXHardwareUnit?,
    request: GenerationRequest,
    onWarning: @escaping @Sendable (String) -> Void
  ) async throws -> Self {
    try request.grammar.validate()
    let start = context.now()
    let directory = try await context.resolveDirectory(source) { repo in
      onWarning("downloading \(repo)...")
    }
    let detection = try context.detectModel(directory)
    let configuration = try EngineRunner.parse(
      request,
      detection: detection,
      requestedEngine: requestedEngine,
      requestedHardwareUnit: hardwareUnit
    )
    let runner = try await context.makeRunner(
      detection,
      configuration.engine,
      configuration.hardwareUnit
    )
    return Self(
      detection: detection,
      runner: runner,
      hardwareUnit: configuration.hardwareUnit,
      loadDuration: start.duration(to: context.now())
    )
  }
}

extension LoadedModel {
  func generate(
    _ request: GenerationRequest,
    onToken: (@Sendable (EdgeToolsToken) -> Void)? = nil,
    onPart: (@Sendable (EdgeToolsGenerationPart) -> Void)? = nil
  ) async throws -> EdgeToolsEngineGeneration {
    if self.runner.engine == .mlx {
      return try await self.hardwareUnit.withDefaultDevice {
        try await self.runner.generate(
          request,
          channel: EdgeToolsGenerationChannel(onToken: onToken, onPart: onPart)
        )
      }
    }
    return try await self.runner.generate(
      request,
      channel: EdgeToolsGenerationChannel(onToken: onToken, onPart: onPart)
    )
  }
}
