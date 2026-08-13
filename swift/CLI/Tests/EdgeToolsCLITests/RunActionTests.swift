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
    expectNoDifference(report.model, "Qwen3")
    expectNoDifference(report.engine, "onnx")
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
      _ = try await runModel(
        context: .stub(engines: [.mlx]),
        source: .test(),
        requestedEngine: .onnx,
        request: GenerationRequest(user: "hello")
      )
    }
  }

  @Test
  func `Uses Model Sampling Defaults Without Request Sampling`() {
    let request = GenerationRequest(user: "hello")

    expectNoDifference(request.sampling.isEmpty, true)
  }

  @Test
  func `Merges Request Sampling Into Model Defaults`() {
    let request = GenerationRequest(
      user: "hello",
      sampling: EdgeToolsFusedSamplingParameters(minP: 0.05, seed: 1234)
    )
    let parameters = request.sampling.applying(
      to: EdgeToolsFusedSamplingParameters(
        temperature: 1,
        topK: 20,
        topP: 0.8,
        repetitionPenalty: 1.1,
        presencePenalty: 2,
        repetitionContextSize: 32
      )
    )

    expectNoDifference(parameters.temperature, 1)
    expectNoDifference(parameters.topK, 20)
    expectNoDifference(parameters.topP, 0.8)
    expectNoDifference(parameters.minP, 0.05)
    expectNoDifference(parameters.repetitionPenalty, 1.1)
    expectNoDifference(parameters.presencePenalty, 2)
    expectNoDifference(parameters.repetitionContextSize, 32)
    expectNoDifference(parameters.seed, 1234)
  }

  @Test
  func `Streams Raw Tokens`() async throws {
    let output = LockedBox("")

    _ = try await runModel(
      context: .stub(runner: .stub(tokens: ["<think>", "answer"])),
      source: .test(),
      request: GenerationRequest(user: "hello"),
      stream: .tokens,
      onOutput: { string, terminator in output.value += string + terminator }
    )

    expectNoDifference(output.value, "<think>answer\n")
  }

  @Test
  func `Throws When Images Are Unsupported By The Engine`() async throws {
    let error = await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(),
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

  @Test
  func `Validates A Custom Grammar Before Resolving The Model`() async throws {
    let resolutions = LockedBox(0)
    let error = await #expect(throws: EdgeCLIError.self) {
      try await runModel(
        context: .stub(onResolve: { _ in resolutions.value += 1 }),
        source: .test(),
        request: GenerationRequest(
          user: "hello",
          grammar: .custom(
            format: .regex,
            source: .file(URL(fileURLWithPath: "/missing/grammar.ebnf"))
          )
        )
      )
    }

    expectNoDifference(resolutions.value, 0)
    expectNoDifference(
      try #require(error).description,
      "Could not read grammar file /missing/grammar.ebnf."
    )
  }
}

private typealias Asset = EdgeToolsTranscript.Asset
