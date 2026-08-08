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
    let report = try await benchmarkModel(
      context: .stub(
        runner: .stub(
          onGenerate: { _ in generations.value += 1 },
          onReset: { resets.value += 1 }
        )
      ),
      source: .test(),
      request: GenerationRequest(user: "hello"),
      runs: 5,
      warmup: 2
    )

    expectNoDifference(report.runs, 5)
    expectNoDifference(report.samples.count, 5)
    expectNoDifference(generations.value, 7)
    expectNoDifference(resets.value, 5)
  }

  @Test
  func `Reports Progress For Each Measured Run`() async throws {
    let progress = LockedBox([Int]())
    let report = try await benchmarkModel(
      context: .stub(),
      source: .test(),
      request: GenerationRequest(user: "hello"),
      runs: 3,
      warmup: 0,
      onProgress: { index, _ in progress.value.append(index) }
    )

    expectNoDifference(report.runs, 3)
    expectNoDifference(progress.value, [1, 2, 3])
  }

  @Test
  func `Counts Runs That Produced Tool Calls`() async throws {
    let call = EdgeRawToolCall(name: "ping", arguments: [:])
    let report = try await benchmarkModel(
      context: .stub(runner: .stub(toolCalls: [call])),
      source: .test(),
      request: GenerationRequest(user: "hello"),
      runs: 4,
      warmup: 0
    )

    expectNoDifference(report.runsWithToolCalls, 4)
  }

  @Test
  func `Propagates Engine Failures`() async {
    await #expect(throws: EdgeCLIError("engine exploded")) {
      try await benchmarkModel(
        context: .stub(runner: .failing(EdgeCLIError("engine exploded"))),
        source: .test(),
        request: GenerationRequest(user: "hello"),
        runs: 2,
        warmup: 0
      )
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
    expectNoDifference(distribution.median, 25)
  }

  @Test
  func `Is Zeroed When There Are No Values`() {
    let distribution = Distribution([])

    expectNoDifference(distribution.median, 0)
    expectNoDifference(distribution.p95, 0)
  }
}
