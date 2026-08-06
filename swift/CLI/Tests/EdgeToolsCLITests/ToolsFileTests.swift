import CustomDump
import EdgeTools
import Foundation
import Testing

@testable import EdgeToolsCLI

@Suite
struct `ToolsFile tests` {
  @Test
  func `Decodes OpenAI Function Calling Format`() throws {
    let json = """
      [
        {"type": "function", "function": {
          "name": "set_timer",
          "description": "Set a timer.",
          "parameters": {"type": "object", "properties": {
            "time_human": {"type": "string"}
          }, "required": ["time_human"]}
        }}
      ]
      """
    let file = try ToolsFile(data: Data(json.utf8))

    expectNoDifference(file.definitions.count, 1)
    expectNoDifference(file.definitions[0].name, "set_timer")
    expectNoDifference(file.definitions[0].description, "Set a timer.")
  }

  @Test
  func `Decodes Bare Tool Objects Using Arguments Key`() throws {
    let json = """
      [{"name": "ping", "description": "Ping.", "arguments": {"type": "object"}}]
      """
    let file = try ToolsFile(data: Data(json.utf8))

    expectNoDifference(file.definitions.map(\.name), ["ping"])
  }

  @Test
  func `Decodes Tools Wrapped In An Object`() throws {
    let json = """
      {"tools": [{"name": "a", "parameters": {"type": "object"}},
                 {"name": "b", "parameters": {"type": "object"}}]}
      """
    let file = try ToolsFile(data: Data(json.utf8))

    expectNoDifference(file.definitions.map(\.name), ["a", "b"])
  }

  @Test
  func `Throws When No Tools Are Defined`() {
    #expect(throws: EdgeCLIError.self) {
      try ToolsFile(data: Data("[]".utf8))
    }
  }

  @Test
  func `Throws When Payload Is Not Tools`() {
    #expect(throws: EdgeCLIError.self) {
      try ToolsFile(data: Data("{\"unrelated\": 1}".utf8))
    }
  }
}
