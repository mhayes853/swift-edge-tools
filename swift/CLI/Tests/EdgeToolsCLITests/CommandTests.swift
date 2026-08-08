import CustomDump
import EdgeTools
import EdgeToolsCLI
import SnapshotTesting
import Testing

@Suite(.snapshots)
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
      arguments: ["--path", "/models/needle", "--stream", "none"],
      context: context
    )

    assertSnapshot(of: capture.standardOutput.value, as: .lines)
    expectNoDifference(capture.standardError.value, "")
    expectNoDifference(try #require(requests.value).user, "Set a timer for five minutes.")
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
        "bench", "--path", "/models/needle", "--prompt", "hello",
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
        engines: [.mlx, .onnx],
        files: ["config.json", "decoder.onnx", "model.safetensors"]
      )
    )

    try await EdgeCommand.run(
      arguments: ["info", "--path", "/models/needle"],
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
          "--path", "/models/needle", "--prompt", "hello",
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
}
