import CustomDump
import EdgeTools
import Testing

@Suite
struct `FunctionGemmaToolCallParser tests` {
  @Test
  func `Parses A Call Incrementally`() throws {
    var parser = FunctionGemmaToolCallParser()
    let chunks = [
      "<start_function_",
      "call>call:get_weather{location:<escape>São ",
      "Paulo<escape>,days:<escape>3<escape>}<end_function_call>"
    ]
    let calls = chunks.enumerated()
      .compactMap { index, chunk in
        parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
      }
    let call = try #require(calls.first)

    expectNoDifference(call.name, "get_weather")
    expectNoDifference(call.arguments, ["location": "São Paulo", "days": 3])
  }
}
