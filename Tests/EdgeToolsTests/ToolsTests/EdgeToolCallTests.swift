import CustomDump
import EdgeTools
import Foundation
import Observation
import Testing

@Suite
struct `EdgeToolCall tests` {
  @Test
  func `Status Starts As Idle`() throws {
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: EchoTool(), rawInput: "hello")
    expectNoDifference(call.status.isIdle, true)
  }

  @Test
  func `Constructs Raw Value From The Tool And Raw Input`() throws {
    let rawInput: EdgeToolsValue = "hello"
    let rawValue = EdgeRawToolCall(name: "echo", arguments: rawInput)
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: EchoTool(), rawInput: rawInput)

    expectNoDifference(call.rawValue, rawValue)
    expectNoDifference(AnyEdgeToolCall(call).rawValue, rawValue)
  }

  @Test
  func `Rejects A Raw Value That Cannot Be Converted To The Tool Input`() {
    #expect(throws: EdgeToolsValueTypeError(expected: .string, received: .integer)) {
      try EdgeToolCall(id: EdgeToolCallID(), tool: EchoTool(), rawInput: 1)
    }
  }

  @Test
  func `Invoke Returns Output And Status Becomes Finished`() async throws {
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: EchoTool(), rawInput: "hello")
    let output = try await call.output

    let expectedOutput = "echo: hello"
    expectNoDifference(output, expectedOutput)
    expectNoDifference(try call.status.result?.get(), expectedOutput)
  }

  @Test
  func `Status Is Running While Tool Is Executing`() async throws {
    let call = try EdgeToolCall(
      id: EdgeToolCallID(),
      tool: DelayedCountingTool(duration: .milliseconds(300), output: "done"),
      rawInput: ""
    )

    _ = Task { try await call.output }

    try await Task.sleep(for: .milliseconds(100))
    expectNoDifference(call.status.isRunning, true)
  }

  @Test
  func `Invoke Deduplicates Concurrent Calls`() async throws {
    let tool = DelayedCountingTool(duration: .milliseconds(200), output: "result")
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "")

    async let firstOutput = try await call.output
    async let secondOutput = try await call.output
    async let thirdOutput = try await call.output

    let results = try await [firstOutput, secondOutput, thirdOutput]

    expectNoDifference(results, ["result", "result", "result"])
    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Invoke Returns Cached Result After Completion`() async throws {
    let tool = DelayedCountingTool(duration: .milliseconds(0), output: "cached")
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "")

    let first = try await call.output
    let second = try await call.output

    expectNoDifference(first, "cached")
    expectNoDifference(second, "cached")
    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Invoke Propagates Errors And Status Becomes Finished With Failure`() async throws {
    let call = try EdgeToolCall(
      id: EdgeToolCallID(),
      tool: ThrowingTool(error: ToolError(message: "boom")),
      rawInput: ""
    )

    await #expect(throws: ToolError.self) {
      _ = try await call.output
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
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "")

    await #expect(throws: ToolError.self) {
      _ = try await call.output
    }
    await #expect(throws: ToolError.self) {
      _ = try await call.output
    }

    expectNoDifference(tool.invokeCount, 1)
  }

  @Test
  func `Cancellation Returns CancellationError`() async throws {
    let call = try EdgeToolCall(
      id: EdgeToolCallID(),
      tool: CancellableTool(duration: .seconds(10)),
      rawInput: ""
    )

    let task = Task { try await call.output }

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
    let call = try EdgeToolCall(id: EdgeToolCallID(), tool: EchoTool(), rawInput: "blob")

    let didChange = Lock(false)
    withObservationTracking {
      _ = call.status
    } onChange: {
      didChange.withLock { $0 = true }
    }

    _ = try await call.output
    didChange.withLock { expectNoDifference($0, true) }
  }
}
