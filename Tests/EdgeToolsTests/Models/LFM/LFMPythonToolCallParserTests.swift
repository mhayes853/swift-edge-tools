import CustomDump
import EdgeTools
import Testing

@Suite
struct `LFMPythonToolCallParser tests` {
  @Test
  func `Parses A Call Incrementally`() throws {
    var parser = LFMPythonToolCallParser()
    let chunks = [
      "<|tool_call_",
      "start|>[get_weather(location='Pa",
      "ris', forecast_days=3)]<|tool_call_end|>"
    ]
    let calls = chunks.enumerated()
      .compactMap { index, chunk in
        parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
      }
    let call = try #require(calls.first)

    expectNoDifference(call.name, "get_weather")
    expectNoDifference(call.arguments, ["location": "Paris", "forecast_days": 3])
  }

  @Test
  func `Decodes Escaped Unicode Surrogate Pairs`() throws {
    var parser = LFMPythonToolCallParser()
    let source = #"<|tool_call_start|>[emoji(value='\uD83D\uDE00')]<|tool_call_end|>"#
    let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
    let call = try #require(parsed)

    expectNoDifference(call.arguments, ["value": "😀"])
  }
}
