import CustomDump
import EdgeTools
import Testing

@Suite
struct `QwenJSONToolCallParser tests` {
  @Test
  func `Decodes Escaped Unicode Surrogate Pairs`() throws {
    var parser = QwenJSONToolCallParser()
    let source = #"<tool_call>{"name":"emoji","arguments":{"value":"\uD83D\uDE00"}}</tool_call>"#
    let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
    let call = try #require(parsed)

    expectNoDifference(call.arguments, ["value": "😀"])
  }
}
