import CustomDump
import EdgeTools
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `RunAction tests` {
  @Test
  func `Reports The Response And Tool Calls From The Engine`() async throws {
    let call = EdgeRawToolCall(name: "set_timer", arguments: ["duration": "5 minutes"])
    let action = RunAction(
      context: .stub(runner: .stub(response: "on it", toolCalls: [call])),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(),
      stream: .none,
      quiet: true
    )
    let report = try await action(prompt: "Set a timer", tools: [])

    expectNoDifference(report.response, "on it")
    expectNoDifference(report.model, "Needle")
    expectNoDifference(report.engine, "mlx")
    expectNoDifference(report.toolCalls.map(\.name), ["set_timer"])
    expectNoDifference(report.metrics.decode.tokens, 2)
  }

  @Test
  func `Passes The Prompt And Tools Through To The Engine`() async throws {
    let requests = LockedBox<GenerationRequest?>(nil)
    let tools = [
      EdgeToolDefinition(name: "ping", description: "Ping.", arguments: [:])
    ]
    let action = RunAction(
      context: .stub(runner: .stub(onGenerate: { requests.value = $0 })),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(toolCallRange: .bounded(1...3)),
      stream: .none,
      quiet: true
    )
    _ = try await action(prompt: "hello", tools: tools)

    let request = try #require(requests.value)
    expectNoDifference(request.user, "hello")
    expectNoDifference(request.tools.map(\.name), ["ping"])
    expectNoDifference(request.toolCallRange, .bounded(1...3))
  }

  @Test
  func `Throws When The Requested Engine Has No Weights`() async {
    let action = RunAction(
      context: .stub(engines: [.mlx]),
      source: .test(),
      requestedEngine: .onnx,
      settings: GenerationSettings(),
      stream: .none,
      quiet: true
    )

    await #expect(throws: EdgeCLIError.self) {
      try await action(prompt: "hello", tools: [])
    }
  }

  @Test
  func `Throws Rather Than Selecting An Experimental Engine`() async throws {
    let action = RunAction(
      context: .stub(engines: [.coreai]),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(),
      stream: .none,
      quiet: true
    )
    let error = await #expect(throws: EdgeCLIError.self) {
      try await action(prompt: "hello", tools: [])
    }

    expectNoDifference(try #require(error).description.contains("--engine: coreai"), true)
  }

  @Test
  func `Throws When A Custom Grammar Is Unsupported By The Engine`() async throws {
    let action = RunAction(
      context: .stub(runner: .stub(supportsCustomGrammar: false)),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(grammar: .unconstrained),
      stream: .none,
      quiet: true
    )
    let error = await #expect(throws: EdgeCLIError.self) {
      try await action(prompt: "hello", tools: [])
    }

    expectNoDifference(try #require(error).description.contains("--grammar auto"), true)
  }

  @Test
  func `Throws When Sampling Is Unsupported By The Engine`() async throws {
    let action = RunAction(
      context: .stub(runner: .stub(supportsSampling: false)),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(temperature: 0.7),
      stream: .none,
      quiet: true
    )
    let error = await #expect(throws: EdgeCLIError.self) {
      try await action(prompt: "hello", tools: [])
    }

    expectNoDifference(try #require(error).description.contains("greedily"), true)
  }
}

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
}
