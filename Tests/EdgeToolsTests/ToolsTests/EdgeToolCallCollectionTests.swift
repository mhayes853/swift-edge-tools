import CustomDump
import Foundation
import EdgeTools
import Testing

@Suite
struct `EdgeToolCallCollection tests` {
  @Test
  func `Typed Mutating APIs Accept Strongly Typed Tool Calls`() {
    var collection = EdgeToolCallCollection()
    let tool = EchoTool()

    collection.append(EdgeToolCall(id: EdgeToolCallID(), tool: tool, input: "a"))
    collection.insert(EdgeToolCall(id: EdgeToolCallID(), tool: tool, input: "b"), at: 0)
    collection.append(contentsOf: [
      EdgeToolCall(id: EdgeToolCallID(), tool: tool, input: "c"),
      EdgeToolCall(id: EdgeToolCallID(), tool: tool, input: "d")
    ])
    collection.insert(
      contentsOf: [EdgeToolCall(id: EdgeToolCallID(), tool: tool, input: "e")],
      at: 1
    )
    collection.replaceSubrange(
      0..<1,
      with: [EdgeToolCall(id: EdgeToolCallID(), tool: tool, input: "head")]
    )

    expectNoDifference(collection.count, 5)
    expectNoDifference(collection[0].tool.name, "echo")
    expectNoDifference(collection[0].input as? String, "head")
  }
}