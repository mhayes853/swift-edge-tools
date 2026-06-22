import CustomDump
import Foundation
import Needle
import Observation
import Testing

@Suite
struct `NeedleToolCall tests` {
  @Test
  func `Status Starts As Idle`() {
    let call = NeedleToolCall(tool: EchoTool(), input: "hello")
    expectNoDifference(call.status.isIdle, true)
  }

  @Test
  func `Invoke Returns Output And Status Becomes Finished`() async throws {
    let call = NeedleToolCall(tool: EchoTool(), input: "hello")
    let output = try await call.invoke()

    let expectedOutput = "echo: hello"
    expectNoDifference(output, expectedOutput)
    expectNoDifference(try call.status.result?.get(), expectedOutput)
  }

  @Test
  func `Status Is Running While Tool Is Executing`() async throws {
    let call = NeedleToolCall(
      tool: CountingTool(duration: .milliseconds(300), output: "done"),
      input: ""
    )

    Task { try await call.invoke() }

    try await Task.sleep(for: .milliseconds(100))
    expectNoDifference(call.status.isRunning, true)
  }

  @Test
  func `Invoke Deduplicates Concurrent Calls`() async throws {
    let tool = CountingTool(duration: .milliseconds(200), output: "result")
    let call = NeedleToolCall(tool: tool, input: "")

    async let a = try await call.invoke()
    async let b = try await call.invoke()
    async let c = try await call.invoke()

    let results = try await [a, b, c]

    expectNoDifference(results, ["result", "result", "result"])
    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Invoke Returns Cached Result After Completion`() async throws {
    let tool = CountingTool(duration: .milliseconds(0), output: "cached")
    let call = NeedleToolCall(tool: tool, input: "")

    let first = try await call.invoke()
    let second = try await call.invoke()

    expectNoDifference(first, "cached")
    expectNoDifference(second, "cached")
    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Invoke Propagates Errors And Status Becomes Finished With Failure`() async throws {
    let call = NeedleToolCall(
      tool: ThrowingTool(error: ToolError(message: "boom")),
      input: ""
    )

    await #expect(throws: ToolError.self) {
      _ = try await call.invoke()
    }
    #expect(throws: ToolError.self) {
      _ = try call.status.result?.get()
    }
  }

  @Test
  func `Cached Error Is Returned On Subsequent Invoke`() async throws {
    let tool = CountingThrowingTool(
      duration: .milliseconds(50),
      error: ToolError(message: "failed")
    )
    let call = NeedleToolCall(tool: tool, input: "")

    await #expect(throws: ToolError.self) {
      _ = try await call.invoke()
    }
    await #expect(throws: ToolError.self) {
      _ = try await call.invoke()
    }

    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Cancellation Returns CancellationError`() async throws {
    let call = NeedleToolCall(
      tool: CancellableTool(duration: .seconds(10)),
      input: ""
    )

    let task = Task { try await call.invoke() }

    try await Task.sleep(for: .milliseconds(50))
    task.cancel()

    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }
    #expect(throws: CancellationError.self) {
      _ = try call.status.result?.get()
    }
  }

  @Test
  func `Status Is Observable`() async throws {
    let call = NeedleToolCall(tool: EchoTool(), input: "blob")

    let didChange = Lock(false)
    withObservationTracking {
      _ = call.status
    } onChange: {
      didChange.withLock { $0 = true }
    }

    _ = try await call.invoke()
    didChange.withLock { expectNoDifference($0, true) }
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

private struct CountingTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "counting"
  let description = "Counts invocations."
  let duration: Duration
  let output: String
  let counter = AtomicCounter()

  var invokeCount: Int { self.counter.value }

  func invoke(input: String) async throws -> sending String {
    self.counter.increment()
    if self.duration > .zero {
      try await Task.sleep(for: self.duration)
    }
    return self.output
  }
}

private struct ThrowingTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "throwing"
  let description = "Always throws."
  let error: any Error

  func invoke(input: String) async throws -> sending String {
    throw self.error
  }
}

private struct CountingThrowingTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "countingThrowing"
  let description = "Counts invocations and always throws."
  let duration: Duration
  let error: any Error
  let counter = AtomicCounter()

  var invokeCount: Int { self.counter.value }

  func invoke(input: String) async throws -> sending String {
    self.counter.increment()
    if self.duration > .zero {
      try await Task.sleep(for: self.duration)
    }
    throw self.error
  }
}

private struct CancellableTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "cancellable"
  let description = "Participates in cooperative cancellation."
  let duration: Duration

  func invoke(input: String) async throws -> sending String {
    try await Task.sleep(for: self.duration)
    return "done"
  }
}

private struct ToolError: Error, Equatable {
  let message: String
}

private final class AtomicCounter: Sendable {
  private let counter = Lock(0)

  var value: Int {
    self.counter.withLock { $0 }
  }

  func increment() {
    self.counter.withLock { $0 += 1 }
  }
}

extension NeedleToolCallStatus {
  var isIdle: Bool {
    switch self {
    case .idle: true
    default: false
    }
  }

  var isRunning: Bool {
    switch self {
    case .running: true
    default: false
    }
  }

  var result: Result<Output, any Error>? {
    switch self {
    case .finished(let result): result
    default: nil
    }
  }
}
