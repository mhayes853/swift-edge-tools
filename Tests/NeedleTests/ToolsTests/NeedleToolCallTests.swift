import CustomDump
import Foundation
import Needle
import Observation
import Testing

@Suite
struct `NeedleToolCall tests` {
  @Test
  func `Status Starts As Idle`() {
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: EchoTool(), input: "hello")
    expectNoDifference(call.status.isIdle, true)
  }

  @Test
  func `Invoke Returns Output And Status Becomes Finished`() async throws {
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: EchoTool(), input: "hello")
    let output = try await call.invokeIfNecessary()

    let expectedOutput = "echo: hello"
    expectNoDifference(output, expectedOutput)
    expectNoDifference(try call.status.result?.get(), expectedOutput)
  }

  @Test
  func `Status Is Running While Tool Is Executing`() async throws {
    let call = NeedleToolCall(
      id: NeedleToolCallID(),
      tool: DelayedCountingTool(duration: .milliseconds(300), output: "done"),
      input: ""
    )

    _ = Task { try await call.invokeIfNecessary() }

    try await Task.sleep(for: .milliseconds(100))
    expectNoDifference(call.status.isRunning, true)
  }

  @Test
  func `Invoke Deduplicates Concurrent Calls`() async throws {
    let tool = DelayedCountingTool(duration: .milliseconds(200), output: "result")
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "")

    async let a = try await call.invokeIfNecessary()
    async let b = try await call.invokeIfNecessary()
    async let c = try await call.invokeIfNecessary()

    let results = try await [a, b, c]

    expectNoDifference(results, ["result", "result", "result"])
    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Invoke Returns Cached Result After Completion`() async throws {
    let tool = DelayedCountingTool(duration: .milliseconds(0), output: "cached")
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "")

    let first = try await call.invokeIfNecessary()
    let second = try await call.invokeIfNecessary()

    expectNoDifference(first, "cached")
    expectNoDifference(second, "cached")
    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Invoke Propagates Errors And Status Becomes Finished With Failure`() async throws {
    let call = NeedleToolCall(
      id: NeedleToolCallID(),
      tool: ThrowingTool(error: ToolError(message: "boom")),
      input: ""
    )

    await #expect(throws: ToolError.self) {
      _ = try await call.invokeIfNecessary()
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
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "")

    await #expect(throws: ToolError.self) {
      _ = try await call.invokeIfNecessary()
    }
    await #expect(throws: ToolError.self) {
      _ = try await call.invokeIfNecessary()
    }

    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Cancellation Returns CancellationError`() async throws {
    let call = NeedleToolCall(
      id: NeedleToolCallID(),
      tool: CancellableTool(duration: .seconds(10)),
      input: ""
    )

    let task = Task { try await call.invokeIfNecessary() }

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
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: EchoTool(), input: "blob")

    let didChange = Lock(false)
    withObservationTracking {
      _ = call.status
    } onChange: {
      didChange.withLock { $0 = true }
    }

    _ = try await call.invokeIfNecessary()
    didChange.withLock { expectNoDifference($0, true) }
  }

  @Test
  func `Input Is Observable`() {
    let call = NeedleToolCall(id: NeedleToolCallID(), tool: EchoTool(), input: "blob")

    let didChange = Lock(false)
    withObservationTracking {
      _ = call.input
    } onChange: {
      didChange.withLock { $0 = true }
    }

    call.input = "new"
    didChange.withLock { expectNoDifference($0, true) }
  }
}
