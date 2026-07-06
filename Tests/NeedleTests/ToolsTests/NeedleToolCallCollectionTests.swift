import CustomDump
import Foundation
import Needle
import Testing

@Suite
struct `NeedleToolCallCollection tests` {
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