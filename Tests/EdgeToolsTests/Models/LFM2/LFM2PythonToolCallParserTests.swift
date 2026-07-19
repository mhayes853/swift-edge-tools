import CustomDump
import EdgeTools
import Testing

@Suite
struct `LFM2PythonToolCallParser tests` {
  @Test
  func `Decodes Escaped Unicode Surrogate Pairs`() throws {
    var parser = LFM2PythonToolCallParser()
    let source = #"<|tool_call_start|>[emoji(value='\uD83D\uDE00')]<|tool_call_end|>"#
    let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
    let call = try #require(parsed)

    expectNoDifference(call.arguments, ["value": "😀"])
  }
}
