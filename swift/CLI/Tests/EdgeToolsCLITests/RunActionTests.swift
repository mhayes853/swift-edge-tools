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
    let report = try await runModel(
      context: .stub(runner: .stub(response: "on it", toolCalls: [call])),
      source: .test(),
      request: GenerationRequest(user: "Set a timer")
    )

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
    _ = try await runModel(
      context: .stub(runner: .stub(onGenerate: { requests.value = $0 })),
      source: .test(),
      request: GenerationRequest(
        user: "hello",
        tools: tools,
        toolCallRange: .bounded(1...3)
      )
    )

    let request = try #require(requests.value)
    expectNoDifference(request.user, "hello")
    expectNoDifference(request.tools.map(\.name), ["ping"])
    expectNoDifference(request.toolCallRange, .bounded(1...3))
  }

  @Test
  func `Throws When The Requested Engine Has No Weights`() async {
    await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(engines: [.mlx]),
        source: .test(),
        requestedEngine: .onnx,
        request: GenerationRequest(user: "hello")
      )
    }
  }

  @Test
  func `Throws Rather Than Selecting An Experimental Engine`() async throws {
    let error = await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(engines: [.coreai]),
        source: .test(),
        request: GenerationRequest(user: "hello")
      )
    }

    expectNoDifference(try #require(error).description.contains("--engine: coreai"), true)
  }

  @Test
  func `Throws When A Custom Grammar Is Unsupported By The Engine`() async throws {
    let error = await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(runner: .stub(supportsCustomGrammar: false)),
        source: .test(),
        request: GenerationRequest(user: "hello", grammar: .unconstrained)
      )
    }

    expectNoDifference(try #require(error).description.contains("--grammar auto"), true)
  }

  @Test
  func `Throws When Sampling Is Unsupported By The Engine`() async throws {
    let error = await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(runner: .stub(supportsSampling: false)),
        source: .test(),
        request: GenerationRequest(user: "hello", temperature: 0.7)
      )
    }

    expectNoDifference(try #require(error).description.contains("greedily"), true)
  }

  @Test
  func `Throws When Images Are Unsupported By The Engine`() async throws {
    let error = await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(runner: .stub(supportsImages: false)),
        source: .test(),
        request: GenerationRequest(user: "hello", images: [Asset(path: "/tmp/cat.png")])
      )
    }

    expectNoDifference(try #require(error).description.contains("--image"), true)
  }

  @Test
  func `Forwards Images To A Vision Engine`() async throws {
    let requests = LockedBox([GenerationRequest]())
    _ = try await runModel(
      context: .stub(
        model: .genericVLM,
        runner: .stub(
          supportsImages: true,
          onGenerate: { request in requests.withValue { $0.append(request) } }
        )
      ),
      source: .test(),
      request: GenerationRequest(user: "hello", images: [Asset(path: "/tmp/cat.png")])
    )

    expectNoDifference(requests.value.first?.images, [Asset(path: "/tmp/cat.png")])
  }
}

private typealias Asset = EdgeToolsLLMPrompt.Asset
