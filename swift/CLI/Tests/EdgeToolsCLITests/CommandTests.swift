import CustomDump
import EdgeTools
import Foundation
import SnapshotTesting
import Testing

@testable import EdgeToolsCLI

@Suite(.serialized, .snapshots)
struct `Command tests` {
  @Test
  func `Run Command E2E`() async throws {
    let requests = LockedBox<GenerationRequest?>(nil)
    let call = EdgeRawToolCall(name: "set_timer", arguments: ["duration": "5 minutes"])
    let (context, capture) = EdgeContext.test(
      context: .stub(
        runner: .stub(
          response: "Timer ready.",
          toolCalls: [call],
          onGenerate: { requests.value = $0 }
        )
      ),
      standardInput: "Set a timer for five minutes.\n"
    )

    try await EdgeCommand.run(
      arguments: [
        "--path", "/models/qwen3", "--stream", "none", "--reasoning", "custom-level"
      ],
      context: context
    )

    assertSnapshot(of: capture.standardOutput.value, as: .lines)
    expectNoDifference(capture.standardError.value, "")
    expectNoDifference(try #require(requests.value).user, "Set a timer for five minutes.")
    expectNoDifference(try #require(requests.value).reasoning.rawValue, "custom-level")
    expectNoDifference(
      try #require(requests.value).tools.map(\.name),
      defaultToolDefinitions.map(\.name)
    )
  }

  @Test
  func `Bench Command E2E`() async throws {
    let (context, capture) = EdgeContext.test()

    try await EdgeCommand.run(
      arguments: [
        "bench", "--path", "/models/qwen3", "--prompt", "hello",
        "--repeat-count", "2", "--warmup", "1"
      ],
      context: context
    )

    assertSnapshot(of: capture.standardOutput.value, as: .lines)
    expectNoDifference(capture.standardError.value, "")
  }

  @Test
  func `Info Command E2E`() async throws {
    let (context, capture) = EdgeContext.test(
      context: .stub(
        engines: [.mlx],
        files: ["config.json", "model.safetensors"]
      )
    )

    try await EdgeCommand.run(
      arguments: ["info", "--path", "/models/qwen3"],
      context: context
    )

    assertSnapshot(of: capture.standardOutput.value, as: .lines)
    expectNoDifference(capture.standardError.value, "")
  }

  @Test
  func `Rejects Maximum Tool Calls Below Minimum Tool Calls`() async {
    let (context, _) = EdgeContext.test()

    do {
      try await EdgeCommand.run(
        arguments: [
          "--path", "/models/qwen3", "--prompt", "hello",
          "--min-tool-calls", "3", "--max-tool-calls", "2"
        ],
        context: context
      )
      Issue.record("Expected command validation to fail.")
    } catch {
      expectNoDifference(
        EdgeCommand.message(for: error),
        "--max-tool-calls must be at least --min-tool-calls."
      )
    }
  }

  @Test
  func `Accepts A Cache Directory For A Hugging Face Repository`() async throws {
    let source = LockedBox<ModelSource?>(nil)
    let (context, _) = EdgeContext.test(
      context: .stub(onResolve: { source.value = $0 })
    )

    try await EdgeCommand.run(
      arguments: [
        "--cache-dir", "/models/cache", "Qwen/Qwen3-0.6B", "--prompt", "hello"
      ],
      context: context
    )

    expectNoDifference(
      try #require(source.value).cacheDirectory,
      URL(fileURLWithPath: "/models/cache")
    )
  }

  @Test
  func `Passes Fused Sampler Options To The Request`() async throws {
    let requests = LockedBox<GenerationRequest?>(nil)
    let (context, _) = EdgeContext.test(
      context: .stub(model: .qwen3, runner: .stub(onGenerate: { requests.value = $0 }))
    )

    try await EdgeCommand.run(
      arguments: [
        "--path", "/models/qwen3", "--prompt", "hello", "--stream", "none",
        "--temperature", "0.7", "--top-k", "40", "--top-p", "0.9", "--min-p", "0.05",
        "--repetition-penalty", "1.1", "--presence-penalty", "0.2",
        "--repetition-context-size", "32", "--seed", "1234"
      ],
      context: context
    )

    let sampling = try #require(requests.value).sampling
    expectNoDifference(sampling.temperature, 0.7)
    expectNoDifference(sampling.topK, 40)
    expectNoDifference(sampling.topP, 0.9)
    expectNoDifference(sampling.minP, 0.05)
    expectNoDifference(sampling.repetitionPenalty, 1.1)
    expectNoDifference(sampling.presencePenalty, 0.2)
    expectNoDifference(sampling.repetitionContextSize, 32)
    expectNoDifference(sampling.seed, 1234)
  }

  @Test
  func `Rejects Negative Maximum Tokens`() async {
    await expectCommandError(
      arguments: ["--path", "/models/qwen3", "--prompt", "hello", "--max-tokens=-1"],
      message: "--max-tokens must be at least 0."
    )
  }

  @Test(arguments: [
    (["--temperature", "nan"], "--temperature must be at least 0."),
    (["--temperature", "inf"], "--temperature must be at least 0."),
    (["--repetition-penalty", "nan"], "--repetition-penalty must be greater than 0."),
    (["--repetition-penalty", "inf"], "--repetition-penalty must be greater than 0."),
    (["--presence-penalty", "nan"], "--presence-penalty must be finite."),
    (["--presence-penalty", "inf"], "--presence-penalty must be finite.")
  ])
  func `Rejects Nonfinite Sampling Values`(options: [String], message: String) async {
    await expectCommandError(
      arguments: ["--path", "/models/qwen3", "--prompt", "hello"] + options,
      message: message
    )
  }

  @Test
  func `Rejects An Empty Top P Candidate Range`() async {
    await expectCommandError(
      arguments: ["--path", "/models/qwen3", "--prompt", "hello", "--top-p", "0"],
      message: "--top-p must be greater than 0 and at most 1."
    )
  }

  @Test
  func `Rejects An Image Directory`() async throws {
    let directory = URL.temporaryDirectory.appending(path: "edge-image-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    await expectCommandError(
      arguments: [
        "--path", "/models/qwen3", "--prompt", "hello", "--image", directory.path()
      ],
      message: "No image file at \(directory.path())."
    )
  }

  @Test
  func `Rejects An Audio Directory`() async throws {
    let directory = URL.temporaryDirectory.appending(path: "edge-audio-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }

    await expectCommandError(
      arguments: [
        "--path", "/models/qwen3", "--prompt", "hello", "--audio", directory.path()
      ],
      message: "No audio file at \(directory.path())."
    )
  }

  @Test
  func `Forwards Repeated Media Options In Order`() async throws {
    let directory = URL.temporaryDirectory.appending(path: "edge-media-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let imagePaths = ["first.png", "second.png"].map { directory.appending(path: $0).path() }
    let audioPaths = ["first.wav", "second.wav"].map { directory.appending(path: $0).path() }
    for path in imagePaths + audioPaths {
      try Data().write(to: URL(filePath: path))
    }

    let requests = LockedBox<GenerationRequest?>(nil)
    let (context, _) = EdgeContext.test(
      context: .multimodalStub { requests.value = $0 }
    )

    try await EdgeCommand.run(
      arguments: [
        "--path", "/models/generic", "--engine", "llama", "--prompt", "hello",
        "--stream", "none", "--image", imagePaths[0], "--image", imagePaths[1],
        "--audio", audioPaths[0], "--audio", audioPaths[1]
      ],
      context: context
    )

    expectNoDifference(
      try #require(requests.value).images.map(\.content),
      imagePaths.map { EdgeToolsTranscript.Asset.Content.path($0) }
    )
    expectNoDifference(
      try #require(requests.value).audio.map(\.content),
      audioPaths.map { EdgeToolsTranscript.Asset.Content.path($0) }
    )
  }

  @Test
  func `Info Does Not Expose Execution Options`() {
    let help = InfoCommand.helpMessage()

    expectNoDifference(help.contains("--path"), true)
    expectNoDifference(help.contains("--engine"), false)
    expectNoDifference(help.contains("--hardware-unit"), false)
  }

  @Test
  func `Clears Benchmark Progress When A Run Fails`() async {
    let (context, capture) = EdgeContext.test(
      context: .stub(runner: .failing(EdgeCLIError("engine exploded"))),
      isStandardErrorTTY: true
    )

    await #expect(throws: EdgeCLIError("engine exploded")) {
      try await EdgeCommand.run(
        arguments: [
          "bench", "--path", "/models/qwen3", "--prompt", "hello", "--warmup", "0"
        ],
        context: context
      )
    }
    expectNoDifference(capture.standardError.value, "run 1/10\r\u{1B}[2K")
  }
}

private func expectCommandError(arguments: [String], message: String) async {
  let (context, _) = EdgeContext.test()
  do {
    try await EdgeCommand.run(arguments: arguments, context: context)
    Issue.record("Expected command validation to fail.")
  } catch {
    expectNoDifference(EdgeCommand.message(for: error), message)
  }
}
