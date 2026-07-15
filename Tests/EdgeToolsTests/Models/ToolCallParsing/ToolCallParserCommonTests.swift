import CustomDump
import EdgeTools
import Testing

@Suite
struct `ToolCallParserCommon tests` {
  @Test
  func `Ignores Text Without Tool Calls`() {
    for fixture in toolCallParserTestFixtures {
      let calls = parseToolCalls(fixture.noCalls, using: fixture.makeParser)

      expectNoDifference(calls, [])
    }
  }

  @Test
  func `Parses Empty Arguments`() throws {
    for fixture in toolCallParserTestFixtures {
      let calls = parseToolCalls(fixture.emptyArguments, using: fixture.makeParser)
      let call = try #require(calls.first)

      expectNoDifference(call, EdgeRawToolCall(name: "empty", arguments: [:]))
    }
  }

  @Test
  func `Parses Complex Arguments`() throws {
    for fixture in toolCallParserTestFixtures {
      let calls = parseToolCalls(fixture.complexCall, using: fixture.makeParser)
      let call = try #require(calls.first)
      guard case .object(let arguments) = call.arguments else {
        Issue.record("Expected object arguments for \(fixture.name).")
        continue
      }

      guard case .object(let destination)? = arguments["destination"] else {
        Issue.record("Expected destination object for \(fixture.name).")
        continue
      }
      guard case .array(let activities)? = arguments["activities"], activities.count == 2 else {
        Issue.record("Expected two activities for \(fixture.name).")
        continue
      }
      guard case .object(let firstActivity) = activities[0],
        case .object(let secondActivity) = activities[1]
      else {
        Issue.record("Expected activity objects for \(fixture.name).")
        continue
      }

      expectNoDifference(call.name, "complex")
      expectNoDifference(destination["city"], "東京")
      expectNoDifference(destination["country"], "日本")
      expectNoDifference(firstActivity["name"], "寿司")
      expectNoDifference(firstActivity["duration"], 2)
      expectNoDifference(secondActivity["name"], "متحف")
      expectNoDifference(secondActivity["duration"], 3)
      expectNoDifference(arguments["tags"], ["e\u{301}", "👩🏽‍💻", "🇺🇳"])
      expectNoDifference(arguments["enabled"], true)
      expectNoDifference(arguments["rating"], 4.5)
      expectNoDifference(arguments["missing"], .null)
      expectNoDifference(arguments["note"], "braces {[]} and \"quotes\" \\ slash\nline")
    }
  }

  @Test
  func `Parses Multiple Calls`() {
    for fixture in toolCallParserTestFixtures {
      let calls = parseToolCalls(fixture.multipleCalls, using: fixture.makeParser)

      expectNoDifference(calls.map(\.name), ["first", "second"])
      expectNoDifference(calls.map(\.arguments), [["value": 1], ["value": 2]])
    }
  }

  @Test
  func `Recovers After A Malformed Call`() {
    for fixture in toolCallParserTestFixtures {
      let calls = parseToolCalls(fixture.malformedThenValid, using: fixture.makeParser)

      expectNoDifference(calls.map(\.name), ["valid"])
      expectNoDifference(calls.map(\.arguments), [["value": 2]])
    }
  }

  @Test
  func `Parses Every Two Way Token Split`() {
    for fixture in toolCallParserTestFixtures {
      let source = fixture.emptyArguments.joined()
      for splitIndex in source.indices.dropFirst() {
        let chunks = [String(source[..<splitIndex]), String(source[splitIndex...])]
        let calls = parseToolCalls(chunks, using: fixture.makeParser)

        expectNoDifference(calls.first?.name, "empty")
      }
    }
  }

  @Test
  func `Preserves Unicode Scalars Across Tokens`() throws {
    for fixture in toolCallParserTestFixtures {
      let calls = parseToolCalls(fixture.unicodeCall, using: fixture.makeParser)
      let call = try #require(calls.first)
      guard case .object(let arguments) = call.arguments else {
        Issue.record("Expected object arguments for \(fixture.name).")
        continue
      }
      guard case .string(let value) = arguments["value"] else {
        Issue.record("Expected a string value for \(fixture.name).")
        continue
      }
      let expected = "e\u{301}👩🏽‍💻 漢字 한글 العربية"

      expectNoDifference(value.unicodeScalars.map(\.value), expected.unicodeScalars.map(\.value))
    }
  }
}
