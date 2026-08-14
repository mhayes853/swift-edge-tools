import CustomDump
import EdgeTools
import Testing

@Suite
struct `GenerationParserCommon tests` {
  @Test
  func `Parses Tool Calls Into Generation Parts`() {
    for fixture in generationParserTestFixtures {
      let parts = parseParts(fixture.source, using: fixture.makeParser)

      expectNoDifference(parts.compactMap(\.toolCall), [fixture.call])
    }
  }

  @Test
  func `Parses Tool Calls Across Every Two Way Token Split`() {
    for fixture in generationParserTestFixtures {
      let source = fixture.source.joined()
      for splitIndex in source.indices.dropFirst() {
        let parts = parseParts(
          [String(source[..<splitIndex]), String(source[splitIndex...])],
          using: fixture.makeParser
        )

        expectNoDifference(parts.compactMap(\.toolCall), [fixture.call])
      }
    }
  }

  @Test
  func `Preserves Text Around Tool Calls`() {
    let parts = parseParts(
      [
        "I will look it up. ",
        #"<tool_call>{"name":"lookup","arguments":{"location":"Paris"}}</tool_call>"#,
        " I will let you know."
      ],
      using: { TestGenerationParser() }
    )

    expectNoDifference(
      parts,
      [
        .text("I will look it up. "),
        .toolCall(EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])),
        .text(" I will let you know.")
      ]
    )
  }

  @Test
  func `Ignores A Qwen XML Closing Tag Inside A Parameter`() throws {
    let parts = parseParts(
      [
        "<tool_call><function=record_note><parameter=text>literal </tool_call> text</parameter></function></tool_call>"
      ],
      using: { Qwen3P5GenerationParser() }
    )
    let call = try #require(parts.compactMap(\.toolCall).first)

    expectNoDifference(call.arguments, ["text": "literal </tool_call> text"])
  }
}

private struct GenerationParserTestFixture: Sendable, CustomStringConvertible {
  let name: String
  let makeParser: @Sendable () -> any EdgeToolsGenerationParser
  let source: [String]
  let call: EdgeRawToolCall

  var description: String { self.name }
}

private let generationParserTestFixtures = [
  GenerationParserTestFixture(
    name: "Qwen3",
    makeParser: { Qwen3GenerationParser() },
    source: [#"<tool_call>{"name":"lookup","arguments":{"location":"Paris"}}</tool_call>"#],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  ),
  GenerationParserTestFixture(
    name: "Qwen3P5",
    makeParser: { Qwen3P5GenerationParser() },
    source: [
      #"<tool_call><function=lookup><parameter=location>"Paris"</parameter></function></tool_call>"#
    ],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  ),
  GenerationParserTestFixture(
    name: "Granite",
    makeParser: { GraniteGenerationParser() },
    source: [#"<tool_call>{"name":"lookup","arguments":{"location":"Paris"}}</tool_call>"#],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  ),
  GenerationParserTestFixture(
    name: "LFM2P5",
    makeParser: { LFM2P5GenerationParser() },
    source: [#"<|tool_call_start|>[lookup(location="Paris")]<|tool_call_end|>"#],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  ),
  GenerationParserTestFixture(
    name: "FunctionGemma",
    makeParser: { FunctionGemmaGenerationParser() },
    source: [
      #"<start_function_call>call:lookup{location:<escape>"Paris"<escape>}<end_function_call>"#
    ],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  ),
  GenerationParserTestFixture(
    name: "Gemma4",
    makeParser: { Gemma4GenerationParser() },
    source: [#"<|tool_call>call:lookup{location:<|"|>Paris<|"|>}<tool_call|>"#],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  ),
  GenerationParserTestFixture(
    name: "MiniCPM5",
    makeParser: { MiniCPM5GenerationParser() },
    source: [#"<function name="lookup"><param name="location">"Paris"</param></function>"#],
    call: EdgeRawToolCall(name: "lookup", arguments: ["location": "Paris"])
  )
]

private func parseParts(
  _ chunks: [String],
  using makeParser: () -> any EdgeToolsGenerationParser
) -> [EdgeToolsGenerationPart] {
  var parser = makeParser()
  return chunks.enumerated()
    .flatMap { index, chunk in
      parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
    } + parser.finish()
}
