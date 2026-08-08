import EdgeTools
import EdgeToolsCLI
import Foundation

final class LockedBox<Value: Sendable>: @unchecked Sendable {
  private let lock = NSLock()
  private var storage: Value

  var value: Value {
    get { self.lock.withLock { self.storage } }
    set { self.lock.withLock { self.storage = newValue } }
  }

  init(_ value: Value) {
    self.storage = value
  }

  func withValue(_ operation: (inout Value) -> Void) {
    self.lock.withLock { operation(&self.storage) }
  }
}

struct CommandCapture: Sendable {
  let standardOutput = LockedBox("")
  let standardError = LockedBox("")
}

extension EdgeContext {
  static func test(
    context: EdgeContext = .stub(),
    standardInput: String = "",
    isStandardErrorTTY: Bool = false
  ) -> (Self, CommandCapture) {
    let capture = CommandCapture()
    return (
      context.withCommandIO(
        standardInput: standardInput,
        capture: capture,
        isStandardErrorTTY: isStandardErrorTTY
      ),
      capture
    )
  }
}

extension EdgeContext {
  static func stub(
    directory: URL = URL(fileURLWithPath: "/models/needle"),
    model: DetectedModel = .needle,
    engines: [EngineKind] = [.mlx],
    files: [String] = ["config.json", "model.safetensors"],
    runner: EngineRunner = .stub(),
    onResolve: @escaping @Sendable (ModelSource) -> Void = { _ in }
  ) -> Self {
    let now = ContinuousClock().now
    return Self(
      resolveDirectory: { source, _ in
        onResolve(source)
        return directory
      },
      detectModel: {
        ModelDetection(directory: $0, model: model, engines: engines, files: files)
      },
      makeRunner: { detection, requestedEngine, hardwareUnit in
        try await EngineRunner(
          detection: detection,
          requestedEngine: requestedEngine,
          hardwareUnit: hardwareUnit,
          loader: { _, _, _ in runner }
        )
      },
      peakMemory: { .zero },
      now: { now },
      readStandardInput: { "" },
      writeStandardOutput: { _, _ in },
      writeStandardError: { _, _ in },
      isStandardErrorTTY: { false }
    )
  }

  private func withCommandIO(
    standardInput: String,
    capture: CommandCapture,
    isStandardErrorTTY: Bool
  ) -> Self {
    var context = self
    context.readStandardInput = { standardInput }
    context.writeStandardOutput = { string, terminator in
      capture.standardOutput.withValue { $0 += string + terminator }
    }
    context.writeStandardError = { string, terminator in
      capture.standardError.withValue { $0 += string + terminator }
    }
    context.isStandardErrorTTY = { isStandardErrorTTY }
    return context
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
