import CustomDump
import EdgeTools
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `BenchAction tests` {
  @Test
  func `Measures Only The Requested Runs And Resets Between Them`() async throws {
    let generations = LockedBox(0)
    let resets = LockedBox(0)
    let action = BenchAction(
      context: .stub(
        runner: .stub(
          onGenerate: { _ in generations.value += 1 },
          onReset: { resets.value += 1 }
        )
      ),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(),
      runs: 5,
      warmup: 2,
      quiet: true
    )
    let report = try await action(prompt: "hello", tools: [])

    expectNoDifference(report.runs, 5)
    expectNoDifference(report.samples.count, 5)
    expectNoDifference(generations.value, 7)
    expectNoDifference(resets.value, 5)
  }

  @Test
  func `Reports Progress For Each Measured Run`() async throws {
    let progress = LockedBox([Int]())
    let action = BenchAction(
      context: .stub(),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(),
      runs: 3,
      warmup: 0,
      quiet: true
    )
    _ = try await action(prompt: "hello", tools: []) { index, _ in
      progress.value.append(index)
    }

    expectNoDifference(progress.value, [1, 2, 3])
  }

  @Test
  func `Counts Runs That Produced Tool Calls`() async throws {
    let call = EdgeRawToolCall(name: "ping", arguments: [:])
    let action = BenchAction(
      context: .stub(runner: .stub(toolCalls: [call])),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(),
      runs: 4,
      warmup: 0,
      quiet: true
    )
    let report = try await action(prompt: "hello", tools: [])

    expectNoDifference(report.runsWithToolCalls, 4)
  }

  @Test
  func `Propagates Engine Failures`() async {
    let action = BenchAction(
      context: .stub(runner: .failing(EdgeCLIError("engine exploded"))),
      source: .test(),
      requestedEngine: nil,
      settings: GenerationSettings(),
      runs: 2,
      warmup: 0,
      quiet: true
    )

    await #expect(throws: EdgeCLIError("engine exploded")) {
      try await action(prompt: "hello", tools: [])
    }
  }
}

@Suite
struct `Distribution tests` {
  @Test
  func `SummarizesValues`() {
    let distribution = Distribution([10, 20, 30, 40])

    expectNoDifference(distribution.min, 10)
    expectNoDifference(distribution.max, 40)
    expectNoDifference(distribution.median, 30)
  }

  @Test
  func `Is Zeroed When There Are No Values`() {
    let distribution = Distribution([])

    expectNoDifference(distribution.median, 0)
    expectNoDifference(distribution.p95, 0)
  }
}
