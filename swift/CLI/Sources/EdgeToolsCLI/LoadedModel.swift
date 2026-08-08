import EdgeTools
import Foundation

// MARK: - LoadedModel

struct LoadedModel {
  var detection: ModelDetection
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
    onWarning: @escaping @Sendable (String) -> Void
  ) async throws -> Self {
    let start = context.now()
    let directory = try await context.resolveDirectory(source) { repo in
      onWarning("downloading \(repo)...")
    }
    let detection = try context.detectModel(directory)
    let runner = try await context.makeRunner(detection, requestedEngine, hardwareUnit)
    if runner.engine.isExperimental {
      onWarning("the \(runner.engine.rawValue) engine is experimental.")
    }
    return Self(
      detection: detection,
      runner: runner,
      loadDuration: start.duration(to: context.now())
    )
  }
}

// MARK: - Requests

extension LoadedModel {
  func validate(_ request: GenerationRequest) throws -> GenerationRequest {
    guard request.temperature == 0 || self.runner.supportsSampling else {
      throw EdgeCLIError(
        """
        \(self.detection.model.displayName) on \(self.runner.engine.rawValue) always samples greedily; \
        --temperature and --top-p do not apply.
        """
      )
    }
    guard request.images.isEmpty || self.runner.supportsImages else {
      throw EdgeCLIError(
        """
        \(self.detection.model.displayName) on \(self.runner.engine.rawValue) takes text only; \
        --image does not apply.
        """
      )
    }
    guard request.grammar == .auto || self.runner.supportsCustomGrammar else {
      throw EdgeCLIError(
        """
        \(self.detection.model.displayName) on \(self.runner.engine.rawValue) only supports \
        `--grammar auto`; its generate parameters expose a tool call range rather than a full \
        generation constraint.
        """
      )
    }
    return request
  }
}

extension LoadedModel {
  func generate(
    _ request: GenerationRequest,
    hardwareUnit: MLXHardwareUnit,
    onToken: (@Sendable (EdgeToolsToken) -> Void)? = nil,
    onToolCall: (@Sendable (EdgeRawToolCall) -> Void)? = nil
  ) async throws -> EdgeToolsEngineGeneration {
    if self.runner.usesMLX {
      return try await hardwareUnit.withDefaultDevice {
        try await self.runner.generate(
          request,
          channel: EdgeToolsGenerationChannel(onToken: onToken, onToolCall: onToolCall)
        )
      }
    }
    return try await self.runner.generate(
      request,
      channel: EdgeToolsGenerationChannel(onToken: onToken, onToolCall: onToolCall)
    )
  }
}
