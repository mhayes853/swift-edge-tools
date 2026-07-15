import CustomDump
import EdgeTools
import Testing

@Suite
struct `QwenXMLToolCallParser tests` {
  @Test
  func `Parses A Call Incrementally`() throws {
    var parser = QwenXMLToolCallParser()
    let chunks = [
      "<tool_call>\n<func",
      "tion=get_weather>\n<parameter=location>\n東",
      "京\n</parameter>\n</function>\n</tool_call>"
    ]
    let calls = chunks.enumerated()
      .compactMap { index, chunk in
        parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
      }
    let call = try #require(calls.first)

    expectNoDifference(call.name, "get_weather")
    expectNoDifference(call.arguments, ["location": "東京"])
  }
}
