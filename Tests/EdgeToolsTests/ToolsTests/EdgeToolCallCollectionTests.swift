import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct `EdgeToolCallCollection tests` {
  @Test
  func `Typed Append Erases Tool Calls`() throws {
    var collection = EdgeToolCallCollection()
    let tool = EchoTool()

    collection.append(try EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: "input"))

    expectNoDifference(collection.count, 1)
    expectNoDifference(collection[0].tool.name, "echo")
    expectNoDifference(collection[0].input as? String, "input")
  }
}
