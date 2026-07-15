import CustomDump
import EdgeTools
import Testing

@Suite
struct `Gemma4ToolCallParser tests` {
  @Test
  func `Parses A Call Incrementally`() throws {
    var parser = Gemma4ToolCallParser()
    let chunks = [
      "<|tool_",
      "call>call:get_weather{location:<|\"|>Zü",
      "rich<|\"|>,days:3}<tool_call|>"
    ]
    let calls = chunks.enumerated()
      .compactMap { index, chunk in
        parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
      }
    let call = try #require(calls.first)

    expectNoDifference(call.name, "get_weather")
    expectNoDifference(call.arguments, ["location": "Zürich", "days": 3])
  }
}
