import CustomDump
import EdgeTools
import Testing

@Suite
struct `QwenJSONToolCallParser tests` {
  @Test
  func `Parses A Call Incrementally`() throws {
    var parser = QwenJSONToolCallParser()
    let chunks = [
      "Reasoning before <tool_",
      "call>\n{\"name\":\"get_weather\",",
      "\"arguments\":{\"location\":\"서울\"}}\n</tool_call>"
    ]
    let calls = chunks.enumerated()
      .compactMap { index, chunk in
        parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
      }
    let call = try #require(calls.first)

    expectNoDifference(call.name, "get_weather")
    expectNoDifference(call.arguments, ["location": "서울"])
  }

  @Test
  func `Decodes Escaped Unicode Surrogate Pairs`() throws {
    var parser = QwenJSONToolCallParser()
    let source = #"<tool_call>{"name":"emoji","arguments":{"value":"\uD83D\uDE00"}}</tool_call>"#
    let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
    let call = try #require(parsed)

    expectNoDifference(call.arguments, ["value": "😀"])
  }
}
