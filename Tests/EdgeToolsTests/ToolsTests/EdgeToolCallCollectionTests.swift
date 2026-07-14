import CustomDump
import Foundation
import EdgeTools
import Testing

@Suite
struct `EdgeToolCallCollection tests` {
  @Test
  func `Typed Mutating APIs Accept Strongly Typed Tool Calls`() throws {
    var collection = EdgeToolCallCollection()
    let tool = EchoTool()

    collection.append(try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "a"))
    collection.insert(try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "b"), at: 0)
    collection.append(contentsOf: [
      try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "c"),
      try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "d")
    ])
    collection.insert(
      contentsOf: [try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "e")],
      at: 1
    )
    collection.replaceSubrange(
      0..<1,
      with: [try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "head")]
    )

    expectNoDifference(collection.count, 5)
    expectNoDifference(collection[0].tool.name, "echo")
    expectNoDifference(collection[0].input as? String, "head")
    expectNoDifference(collection[0].rawValue.arguments, "head")
  }
}
