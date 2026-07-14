import CustomDump
import EdgeTools
import Testing

@Suite
struct `NeedleToolCallParser tests` {
  @Test
  func `Parses A Call Incrementally`() throws {
    let calls = self.parse([
      "<tool_",
      "call> [{\"name\":\"get_weather\",",
      "\"arguments\":{\"location\":\"Seoul\"}}]"
    ])

    let call = try #require(calls.first)
    expectNoDifference(call.name, "get_weather")
    expectNoDifference(call.arguments, ["location": "Seoul"])
  }

  @Test
  func `Parses Nested Objects And Arrays`() throws {
    let calls = self.parse([
      #"<tool_call> [{"name":"plan_trip","arguments":{"destination":{"city":"Tokyo","country":"Japan"},"#,
      #""activities":[{"name":"Sushi","duration":2},{"name":"Temple","duration":3}],"#,
      #""tags":["food","culture"]}}]"#
    ])

    let call = try #require(calls.first)
    guard case .object(let arguments) = call.arguments else {
      Issue.record("Expected object arguments.")
      return
    }
    let destinationValue = try #require(arguments["destination"])
    guard case .object(let destination) = destinationValue else {
      Issue.record("Expected object destination.")
      return
    }
    let activitiesValue = try #require(arguments["activities"])
    guard case .array(let activities) = activitiesValue else {
      Issue.record("Expected array activities.")
      return
    }
    let tags = try #require(arguments["tags"])

    expectNoDifference(call.name, "plan_trip")
    expectNoDifference(destination["city"], "Tokyo")
    expectNoDifference(destination["country"], "Japan")
    expectNoDifference(activities.count, 2)
    expectNoDifference(tags, ["food", "culture"])
  }

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
  func `Parses Multiple Calls`() throws {
    let calls = self.parse([
      #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},"#,
      #"{"name":"get_weather","arguments":{"location":"Paris"}}]"#
    ])

    expectNoDifference(calls.map(\.name), ["get_weather", "get_weather"])
    expectNoDifference(calls.map(\.arguments), [
      ["location": "Seoul"],
      ["location": "Paris"]
    ])
  }

  @Test
  func `Ignores Malformed Calls`() {
    let calls = self.parse([
      #"<tool_call> [{"name":"get_weather","arguments":{"location":}}]"#
    ])

    expectNoDifference(calls, [])
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
