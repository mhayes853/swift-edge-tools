import CustomDump
import EdgeTools
import Testing

@Suite
struct `NeedleToolCallParser tests` {
  @Test
  func `Ignores Boundaries Inside JSON Strings`() throws {
    let calls = self.parse([
      #"<tool_call> [{"name":"record_note","arguments":{"text":"literal}_and]_inside_string","#,
      #""title":"{draft}"}}]"#
    ])

    let call = try #require(calls.first)
    guard case .object(let arguments) = call.arguments else {
      Issue.record("Expected object arguments.")
      return
    }

    expectNoDifference(arguments["text"], "literal}_and]_inside_string")
    expectNoDifference(arguments["title"], "{draft}")
  }

  @Test
  func `Preserves The Raw Tool Name`() throws {
    let calls = self.parse([
      #"<tool_call> [{"name":"getWeather","arguments":{"location":"Seoul"}}]"#
    ])
    let call = try #require(calls.first)

    expectNoDifference(call.name, "getWeather")
  }

  private func parse(_ chunks: [String]) -> [EdgeRawToolCall] {
    var parser = NeedleToolCallParser()
    var calls = [EdgeRawToolCall]()
    for (index, chunk) in chunks.enumerated() {
      let token = EdgeToolsToken(id: index, stringValue: chunk)
      if let call = parser.accept(token: token) {
        calls.append(call)
      }
    }
    return calls
  }
}
