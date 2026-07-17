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

struct ToolCallParserTestFixture: Sendable, CustomStringConvertible {
  let name: String
  let makeParser: @Sendable () -> any EdgeToolCallParser
  let noCalls: [String]
  let emptyArguments: [String]
  let complexCall: [String]
  let multipleCalls: [String]
  let malformedThenValid: [String]
  let unicodeCall: [String]

  var description: String { self.name }
}

let toolCallParserTestFixtures = [
  ToolCallParserTestFixture(
    name: "Needle",
    makeParser: { NeedleToolCallParser() },
    noCalls: ["There are no tools to call."],
    emptyArguments: [#"<tool_call> [{"name":"empty","arguments":{}}]"#],
    complexCall: [
      #"<tool_call> [{"name":"complex","arguments":{"destination":{"city":"東京","country":"日本"},"activities":["#,
      #"{"name":"寿司","duration":2},{"name":"متحف","duration":3}],"tags":["#,
      #""e\u0301","👩🏽‍💻","🇺🇳"],"enabled":true,"rating":4.5,"missing":null,"#,
      #""note":"braces {[]} and \"quotes\" \\ slash\nline"}}]"#
    ],
    multipleCalls: [
      #"<tool_call> [{"name":"first","arguments":{"value":1}},"#,
      #"{"name":"second","arguments":{"value":2}}]"#
    ],
    malformedThenValid: [
      #"<tool_call> [{"name":"bad","arguments":{"value":}},"#,
      #"{"name":"valid","arguments":{"value":2}}]"#
    ],
    unicodeCall: [
      #"<tool_call> [{"name":"unicode","arguments":{"value":"e"#,
      "\u{301}",
      "👩🏽",
      "\u{200D}",
      #"💻 漢字 한글 العربية"}}]"#
    ]
  ),
  ToolCallParserTestFixture(
    name: "Qwen JSON",
    makeParser: { QwenJSONToolCallParser() },
    noCalls: ["There are no tools to call."],
    emptyArguments: [#"<tool_call>{"name":"empty","arguments":{}}</tool_call>"#],
    complexCall: [
      #"<tool_call>{"name":"complex","arguments":{"destination":{"city":"東京","country":"日本"},"activities":["#,
      #"{"name":"寿司","duration":2},{"name":"متحف","duration":3}],"tags":["#,
      #""e\u0301","👩🏽‍💻","🇺🇳"],"enabled":true,"rating":4.5,"missing":null,"#,
      #""note":"braces {[]} and \"quotes\" \\ slash\nline"}}</tool_call>"#
    ],
    multipleCalls: [
      #"<tool_call>{"name":"first","arguments":{"value":1}}</tool_call>"#,
      #"<tool_call>{"name":"second","arguments":{"value":2}}</tool_call>"#
    ],
    malformedThenValid: [
      #"<tool_call>{"name":"bad","arguments":{"value":}}</tool_call>"#,
      #"<tool_call>{"name":"valid","arguments":{"value":2}}</tool_call>"#
    ],
    unicodeCall: [
      #"<tool_call>{"name":"unicode","arguments":{"value":"e"#,
      "\u{301}",
      "👩🏽",
      "\u{200D}",
      #"💻 漢字 한글 العربية"}}</tool_call>"#
    ]
  ),
  ToolCallParserTestFixture(
    name: "Qwen XML",
    makeParser: { QwenXMLToolCallParser() },
    noCalls: ["There are no tools to call."],
    emptyArguments: ["<tool_call><function=empty></function></tool_call>"],
    complexCall: [
      #"""
      <tool_call><function=complex>
      <parameter=destination>{"city":"東京","country":"日本"}</parameter>
      <parameter=activities>[{"name":"寿司","duration":2},{"name":"متحف","duration":3}]</parameter>
      <parameter=tags>["e\u0301","👩🏽‍💻","🇺🇳"]</parameter>
      <parameter=enabled>True</parameter><parameter=rating>4.5</parameter>
      <parameter=missing>None</parameter>
      <parameter=note>braces {[]} and "quotes" \ slash
      line</parameter></function></tool_call>
      """#
    ],
    multipleCalls: [
      "<tool_call><function=first><parameter=value>1</parameter></function></tool_call>",
      "<tool_call><function=second><parameter=value>2</parameter></function></tool_call>"
    ],
    malformedThenValid: [
      "<tool_call><function=bad><parameter=value>1</function></tool_call>",
      "<tool_call><function=valid><parameter=value>2</parameter></function></tool_call>"
    ],
    unicodeCall: [
      "<tool_call><function=unicode><parameter=value>e",
      "\u{301}",
      "👩🏽",
      "\u{200D}",
      "💻 漢字 한글 العربية</parameter></function></tool_call>"
    ]
  ),
  ToolCallParserTestFixture(
    name: "LFM Python",
    makeParser: { LFMPythonToolCallParser() },
    noCalls: ["There are no tools to call."],
    emptyArguments: ["<|tool_call_start|>[empty()]<|tool_call_end|>"],
    complexCall: [
      #"""
      <|tool_call_start|>[complex(
        destination={'city':'東京','country':'日本'},
        activities=[{'name':'寿司','duration':2},{'name':'متحف','duration':3}],
        tags=['e\u0301','👩🏽‍💻','🇺🇳'], enabled=True, rating=4.5, missing=None,
        note='braces {[]} and "quotes" \\ slash\nline'
      )]<|tool_call_end|>
      """#
    ],
    multipleCalls: [
      "<|tool_call_start|>[first(value=1),",
      "second(value=2)]<|tool_call_end|>"
    ],
    malformedThenValid: [
      "<|tool_call_start|>[bad(value=),",
      "valid(value=2)]<|tool_call_end|>"
    ],
    unicodeCall: [
      "<|tool_call_start|>[unicode(value='e",
      "\u{301}",
      "👩🏽",
      "\u{200D}",
      "💻 漢字 한글 العربية')]<|tool_call_end|>"
    ]
  ),
  ToolCallParserTestFixture(
    name: "FunctionGemma",
    makeParser: { FunctionGemmaToolCallParser() },
    noCalls: ["There are no tools to call."],
    emptyArguments: ["<start_function_call>call:empty{}<end_function_call>"],
    complexCall: [
      #"""
      <start_function_call>call:complex{
      destination:<escape>{"city":"東京","country":"日本"}<escape>,
      activities:<escape>[{"name":"寿司","duration":2},{"name":"متحف","duration":3}]<escape>,
      tags:<escape>["é","👩🏽‍💻","🇺🇳"]<escape>,
      enabled:<escape>true<escape>,rating:<escape>4.5<escape>,missing:<escape>null<escape>,
      note:<escape>braces {[]} and "quotes" \ slash
      line<escape>}<end_function_call>
      """#
    ],
    multipleCalls: [
      "<start_function_call>call:first{value:<escape>1<escape>}<end_function_call>",
      "<start_function_call>call:second{value:<escape>2<escape>}<end_function_call>"
    ],
    malformedThenValid: [
      "<start_function_call>call:bad{value:}<end_function_call>",
      "<start_function_call>call:valid{value:<escape>2<escape>}<end_function_call>"
    ],
    unicodeCall: [
      "<start_function_call>call:unicode{value:<escape>e",
      "\u{301}",
      "👩🏽",
      "\u{200D}",
      "💻 漢字 한글 العربية<escape>}<end_function_call>"
    ]
  ),
  ToolCallParserTestFixture(
    name: "Gemma4",
    makeParser: { Gemma4ToolCallParser() },
    noCalls: ["There are no tools to call."],
    emptyArguments: ["<|tool_call>call:empty{}<tool_call|>"],
    complexCall: [
      #"""
      <|tool_call>call:complex{
      destination:{city:<|"|>東京<|"|>,country:<|"|>日本<|"|>},
      activities:[{name:<|"|>寿司<|"|>,duration:2},{name:<|"|>متحف<|"|>,duration:3}],
      tags:[<|"|>é<|"|>,<|"|>👩🏽‍💻<|"|>,<|"|>🇺🇳<|"|>],
      enabled:true,rating:4.5,missing:null,
      note:<|"|>braces {[]} and "quotes" \ slash
      line<|"|>}<tool_call|>
      """#
    ],
    multipleCalls: [
      "<|tool_call>call:first{value:1}<tool_call|>",
      "<|tool_call>call:second{value:2}<tool_call|>"
    ],
    malformedThenValid: [
      "<|tool_call>call:bad{value:}<tool_call|>",
      "<|tool_call>call:valid{value:2}<tool_call|>"
    ],
    unicodeCall: [
      "<|tool_call>call:unicode{value:<|\"|>e",
      "\u{301}",
      "👩🏽",
      "\u{200D}",
      "💻 漢字 한글 العربية<|\"|>}<tool_call|>"
    ]
  )
]

func parseToolCalls(
  _ chunks: [String],
  using makeParser: () -> any EdgeToolCallParser
) -> [EdgeRawToolCall] {
  var parser = makeParser()
  return chunks.enumerated()
    .compactMap { index, chunk in
      parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
    }
}
