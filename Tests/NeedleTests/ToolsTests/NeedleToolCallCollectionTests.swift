import CustomDump
import Foundation
import Needle
import Observation
import Testing

@Suite
struct `NeedleToolCallCollection tests` {
  @Test
  func `InvokeAll Invokes Every Tool`() async throws {
    var collection = NeedleToolCallCollection()
    let call1 = AnyNeedleToolCall(NeedleToolCall(tool: EchoTool(), input: "a"))
    let call2 = AnyNeedleToolCall(NeedleToolCall(tool: EchoTool(), input: "b"))
    collection.append(call1)
    collection.append(call2)

    try await collection.invokeAll()

    expectNoDifference(call1.status.isFinished, true)
    expectNoDifference(call2.status.isFinished, true)
  }

  @Test
  func `InvokeAll Where Invokes Only Matching Tools`() async throws {
    var collection = NeedleToolCallCollection()
    let matchingTool = CountingTool(name: "match", output: "yes")
    let nonMatchingTool = CountingTool(name: "skip", output: "no")
    let matchingCall = AnyNeedleToolCall(NeedleToolCall(tool: matchingTool, input: ""))
    let nonMatchingCall = AnyNeedleToolCall(NeedleToolCall(tool: nonMatchingTool, input: ""))
    collection.append(matchingCall)
    collection.append(nonMatchingCall)

    try await collection.invokeAll(where: { $0.tool.name == "match" })

    expectNoDifference(matchingTool.invokeCount, 1)
    expectNoDifference(nonMatchingTool.invokeCount, 0)
  }

  @Test
  func `InvokeAll Throws Aggregate Error When Some Tools Fail`() async throws {
    var collection = NeedleToolCallCollection()
    let successTool = CountingTool(name: "success", output: "ok")
    let failureTool = ThrowingTool(name: "failure", error: CollectionTestError(message: "boom"))
    let failureCall = AnyNeedleToolCall(NeedleToolCall(tool: failureTool, input: ""))
    let successCall = AnyNeedleToolCall(NeedleToolCall(tool: successTool, input: ""))
    collection.append(failureCall)
    collection.append(successCall)

    let error = await #expect(throws: NeedleToolCallCollection.InvocationError.self) {
      try await collection.invokeAll()
    }

    expectNoDifference(error?.failures.count, 1)
    expectNoDifference(error?.failures.first?.toolCall.tool.name, "failure")
    expectNoDifference(successTool.invokeCount, 1)
  }

  @Test
  func `InvokeAll Does Not Throw When No Tools Match`() async throws {
    var collection = NeedleToolCallCollection()
    let tool = CountingTool(name: "skip", output: "no")
    collection.append(AnyNeedleToolCall(NeedleToolCall(tool: tool, input: "")))

    try await collection.invokeAll(where: { _ in false })

    expectNoDifference(tool.invokeCount, 0)
  }

  @Test
  func `InvokeAll Does Not Throw On Empty Collection`() async throws {
    let collection = NeedleToolCallCollection()

    try await collection.invokeAll()
  }

  @Test
  func `InvokeAll Runs Tools In Parallel`() async throws {
    let probe = ParallelProbe()
    var collection = NeedleToolCallCollection()
    collection.append(AnyNeedleToolCall(NeedleToolCall(tool: ProbeTool(probe: probe), input: "")))
    collection.append(AnyNeedleToolCall(NeedleToolCall(tool: ProbeTool(probe: probe), input: "")))

    try await collection.invokeAll()

    expectNoDifference(probe.maxConcurrent, 2)
  }
}

// MARK: - Helpers

private struct EchoTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "echo"
  let description = "Returns a fixed string."

  func invoke(input: String) async throws -> sending String {
    "echo: \(input)"
  }
}

private final class CountingTool: NeedleTool, Sendable {
  typealias Input = String
  typealias Output = String

  let name: String
  let description = "Counts invocations."
  let output: String
  private let counter = Lock(0)

  init(name: String, output: String) {
    self.name = name
    self.output = output
  }

  var invokeCount: Int {
    self.counter.withLock { $0 }
  }

  func invoke(input: String) async throws -> sending String {
    self.counter.withLock { $0 += 1 }
    return self.output
  }
}

private struct ThrowingTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name: String
  let description = "Always throws."
  let error: any Error

  func invoke(input: String) async throws -> sending String {
    throw self.error
  }
}

private struct CollectionTestError: Error, Equatable {
  let message: String
}

private final class ParallelProbe: Sendable {
  private let state = Lock((current: 0, max: 0))

  func enter() {
    self.state.withLock { state in
      state.current += 1
      state.max = Swift.max(state.max, state.current)
    }
  }

  func exit() {
    self.state.withLock { $0.current -= 1 }
  }

  var maxConcurrent: Int {
    self.state.withLock { $0.max }
  }
}

private struct ProbeTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "probe"
  let description = "Tracks concurrency."
  let probe: ParallelProbe

  func invoke(input: String) async throws -> sending String {
    self.probe.enter()
    try await Task.sleep(for: .milliseconds(50))
    self.probe.exit()
    return "done"
  }
}

extension NeedleToolCallStatus {
  fileprivate var isFinished: Bool {
    switch self {
    case .finished: true
    default: false
    }
  }
}
