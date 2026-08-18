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
    expectNoDifference(report.engine, "mlx")
    expectNoDifference(report.toolCalls.map(\.name), ["set_timer"])
    expectNoDifference(report.metrics.generation["decodeTokens"], 2)
  }

  @Test
  func `Uses The Engines Metric Extractor For User Facing Reports`() async throws {
    var metadata = EdgeToolsMetadata()
    metadata.needle2PrefillTokensPerSecond = 847.5
    metadata.needle2DecodeTokensPerSecond = 96.25
    metadata.needle2PeakRAMMegabytes = 31.75
    let report = try await runModel(
      context: .stub(
        runner: .stub(
          metadata: metadata,
          metricsExtractor: Needle2GenerationMetricsExtractor()
        )
      ),
      source: .test(),
      request: GenerationRequest(user: "hello")
    )

    expectNoDifference(
      report.metrics.generation.groups.map(\.label),
      ["Prefill", "Decode", "RAM"]
    )
    expectNoDifference(report.metrics.generation["prefillTokensPerSecond"], 847.5)
    expectNoDifference(report.metrics.generation["decodeTokensPerSecond"], 96.25)
    expectNoDifference(report.metrics.generation["timeToFirstTokenMilliseconds"], nil)
    expectNoDifference(report.metrics.generation["needle2PeakRAMMegabytes"], 31.75)
  }

  @Test
  func `Reports Generation And Probe Confidence As Percentages`() async throws {
    var metadata = EdgeToolsMetadata()
    metadata.generationConfidence = 0.8125
    metadata.probeConfidence = 0.5
    let report = try await runModel(
      context: .stub(runner: .stub(metadata: metadata)),
      source: .test(),
      request: GenerationRequest(user: "hello")
    )

    expectNoDifference(report.metrics.generation["generationConfidence"], 81.25)
    expectNoDifference(report.metrics.generation["probeConfidence"], 50)
    expectNoDifference(
      report.displayText(includingResponse: false).contains("Confidence 81.2%  probe 50.0%"),
      true
    )
  }

  @Test
  func `Omits The Confidence Group When The Engine Scores Nothing`() async throws {
    let report = try await runModel(
      context: .stub(runner: .stub()),
      source: .test(),
      request: GenerationRequest(user: "hello")
    )

    expectNoDifference(report.metrics.generation.groups.map(\.label), ["Prefill", "Decode"])
  }

  @Test
  func `Omits Probe Confidence When The Model Carries No Probe`() async throws {
    var metadata = EdgeToolsMetadata()
    metadata.generationConfidence = 0.5
    let report = try await runModel(
      context: .stub(runner: .stub(metadata: metadata)),
      source: .test(),
      request: GenerationRequest(user: "hello")
    )

    expectNoDifference(report.metrics.generation["probeConfidence"], nil)
    expectNoDifference(report.metrics.generation.groups.map(\.label).contains("Confidence"), true)
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
