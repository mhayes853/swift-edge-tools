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
    let call1 = NeedleToolCall(id: NeedleToolCallID(), tool: EchoTool(), input: "a")
    let call2 = NeedleToolCall(id: NeedleToolCallID(), tool: EchoTool(), input: "b")
    collection.append(call1)
    collection.append(call2)

    let results = await collection.invokeAllIfNecessary()

    expectNoDifference(results.count, 2)
    expectNoDifference(call1.status.isFinished, true)
    expectNoDifference(call2.status.isFinished, true)
  }

  @Test
  func `InvokeAll Where Invokes Only Matching Tools`() async throws {
    var collection = NeedleToolCallCollection()
    let matchingTool = CountingTool(name: "match", output: "yes")
    let nonMatchingTool = CountingTool(name: "skip", output: "no")
    let matchingCall = NeedleToolCall(id: NeedleToolCallID(), tool: matchingTool, input: "")
    let nonMatchingCall = NeedleToolCall(id: NeedleToolCallID(), tool: nonMatchingTool, input: "")
    collection.append(matchingCall)
    collection.append(nonMatchingCall)

    let results = await collection.invokeAllIfNecessary(where: { $0.tool.name == "match" })

    expectNoDifference(results.count, 1)
    expectNoDifference(results.first?.tool.name, "match")
    expectNoDifference(matchingTool.invokeCount, 1)
    expectNoDifference(nonMatchingTool.invokeCount, 0)
  }

  @Test
  func `InvokeAll Returns Mixed Results When Some Tools Fail`() async throws {
    var collection = NeedleToolCallCollection()
    let successTool = CountingTool(name: "success", output: "ok")
    let failureTool = ThrowingTool(name: "failure", error: ToolError(message: "boom"))
    let failureCall = NeedleToolCall(id: NeedleToolCallID(), tool: failureTool, input: "")
    let successCall = NeedleToolCall(id: NeedleToolCallID(), tool: successTool, input: "")
    collection.append(failureCall)
    collection.append(successCall)

    let results = await collection.invokeAllIfNecessary()

    expectNoDifference(results.count, 2)
    let failures = results.filter { $0.tool.name == "failure" }
    let successes = results.filter { $0.tool.name == "success" }
    expectNoDifference(failures.count, 1)
    expectNoDifference(successes.count, 1)
    let failureError: ToolError = {
      guard let result = failures.first?.result else { return ToolError(message: "missing") }
      if case .failure(let error) = result, let cast = error as? ToolError {
        return cast
      }
      return ToolError(message: "unexpected success")
    }()
    expectNoDifference(failureError.message, "boom")
    expectNoDifference(successTool.invokeCount, 1)
  }

  @Test
  func `InvokeAll Does Not Invoke When No Tools Match`() async throws {
    var collection = NeedleToolCallCollection()
    let tool = CountingTool(name: "skip", output: "no")
    collection.append(NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: ""))

    let results = await collection.invokeAllIfNecessary(where: { _ in false })

    expectNoDifference(results.count, 0)
    expectNoDifference(tool.invokeCount, 0)
  }

  @Test
  func `InvokeAll Returns Empty Array For Empty Collection`() async throws {
    let collection = NeedleToolCallCollection()

    let results = await collection.invokeAllIfNecessary()

    expectNoDifference(results.count, 0)
  }

  @Test
  func `InvokeAll Runs Tools In Parallel`() async throws {
    let probe = ParallelProbe()
    var collection = NeedleToolCallCollection()
    collection.append(
      NeedleToolCall(id: NeedleToolCallID(), tool: ProbeTool(probe: probe), input: "")
    )
    collection.append(
      NeedleToolCall(id: NeedleToolCallID(), tool: ProbeTool(probe: probe), input: "")
    )

    _ = await collection.invokeAllIfNecessary()

    expectNoDifference(probe.maxConcurrent, 2)
  }

  @Test
  func `Typed Mutating APIs Accept Strongly Typed Tool Calls`() {
    var collection = NeedleToolCallCollection()
    let tool = EchoTool()

    collection.append(NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "a"))
    collection.insert(NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "b"), at: 0)
    collection.append(contentsOf: [
      NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "c"),
      NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "d")
    ])
    collection.insert(
      contentsOf: [NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "e")],
      at: 1
    )
    collection.replaceSubrange(
      0..<1,
      with: [NeedleToolCall(id: NeedleToolCallID(), tool: tool, input: "head")]
    )

    expectNoDifference(collection.count, 5)
    expectNoDifference(collection[0].tool.name, "echo")
    expectNoDifference(collection[0].input as? String, "head")
  }
}
