import EdgeTools
import EdgeToolsCLI
import Foundation

extension EdgeContext {
  static func stub(
    directory: URL = URL(fileURLWithPath: "/models/needle"),
    model: DetectedModel = .needle,
    engines: [EngineKind] = [.mlx],
    files: [String] = ["config.json", "model.safetensors"],
    runner: EngineRunner = .stub(),
    onResolve: @escaping @Sendable (ModelSource) -> Void = { _ in }
  ) -> Self {
    Self(
      resolveDirectory: { source, _ in
        onResolve(source)
        return directory
      },
      detectModel: {
        ModelDetection(directory: $0, model: model, engines: engines, files: files)
      },
      makeRunner: { _, _ in runner },
      peakMemory: { .zero }
    )
  }
}

extension EngineRunner {
  static func stub(
    response: String = "done",
    toolCalls: [EdgeRawToolCall] = [],
    tokens: [String] = ["do", "ne"],
    supportsCustomGrammar: Bool = true,
    supportsSampling: Bool = true,
    decodeDuration: Duration = .milliseconds(20),
    onGenerate: @escaping @Sendable (GenerationRequest) -> Void = { _ in },
    onReset: @escaping @Sendable () -> Void = {}
  ) -> Self {
    Self(
      supportsCustomGrammar: supportsCustomGrammar,
      supportsSampling: supportsSampling,
      generation: { request, channel in
        onGenerate(request)
        for (index, token) in tokens.enumerated() {
          channel.emit(token: EdgeToolsToken(id: index, stringValue: token))
        }
        for call in toolCalls {
          channel.emit(toolCall: call)
        }
        return EdgeToolsEngineGeneration(
          prefillMetrics: EdgeToolsPrefillMetrics(tokens: 10, duration: .milliseconds(5)),
          decodeMetrics: EdgeToolsDecodeMetrics(
            tokens: tokens.count,
            duration: decodeDuration,
            durationToFirstToken: .milliseconds(2)
          ),
          wasStopped: false,
          tokens: tokens.enumerated()
            .map { EdgeToolsToken(id: $0.offset, stringValue: $0.element) },
          response: response,
          toolCalls: toolCalls
        )
      },
      cacheClearing: { onReset() }
    )
  }

  static func failing(_ error: any Error) -> Self {
    Self(
      supportsCustomGrammar: true,
      supportsSampling: true,
      generation: { _, _ in throw error }
    )
  }
}

extension ModelSource {
  static func test(repo: String = "Cactus-Compute/needle") -> Self {
    Self(location: .huggingFace(repo: repo, revision: "main"))
  }
}
