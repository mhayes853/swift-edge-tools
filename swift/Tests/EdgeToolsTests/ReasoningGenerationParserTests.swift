import EdgeTools
import Testing

@Suite
struct `ReasoningGenerationParser tests` {
  @Test
  func `Separates Qwen3 Reasoning From Text`() {
    var parser = Qwen3GenerationParser()
    assertReasoningParts(
      parser: &parser,
      chunks: ["<think>", "check the weather", "</think>", "Sunny."]
    )
  }

  @Test
  func `Separates Qwen3P5 Reasoning From Text`() {
    var parser = Qwen3P5GenerationParser()
    assertReasoningParts(
      parser: &parser,
      prefix: "<think>\n",
      chunks: ["check the weather", "</think>", "Sunny."]
    )
  }

  @Test
  func `Separates MiniCPM5 Reasoning From Text`() {
    var parser = MiniCPM5GenerationParser()
    assertReasoningParts(
      parser: &parser,
      prefix: "<think>\n",
      chunks: ["check the weather", "</think>", "Sunny."]
    )
  }

  @Test
  func `Qwen3P5 Skips Its Empty Reasoning Region When Disabled`() {
    var parser = Qwen3P5GenerationParser()
    assertTextParts(
      parser: &parser,
      prefix: "<think>\n\n</think>\n\n",
      chunks: ["Sunny."]
    )
  }

  @Test
  func `MiniCPM5 Skips Its Empty Reasoning Region When Disabled`() {
    var parser = MiniCPM5GenerationParser()
    assertTextParts(
      parser: &parser,
      prefix: "<think>\n\n</think>\n\n",
      chunks: ["Sunny."]
    )
  }

  @Test
  func `Separates Gemma4 Reasoning From Text`() {
    var parser = Gemma4GenerationParser()
    assertReasoningParts(
      parser: &parser,
      chunks: ["<|channel>thought\n", "check the weather", "<channel|>", "Sunny."]
    )
  }

  @Test
  func `Separates LFM2P5 Reasoning From Text`() {
    var parser = LFM2P5GenerationParser()
    assertReasoningParts(
      parser: &parser,
      chunks: ["<think>", "check the weather", "</think>", "Sunny."]
    )
  }

  @Test
  func `Parses Qwen Reasoning Before A Tool Call`() {
    var parser = Qwen3GenerationParser()
    let parts =
      [
        "<think>", "I need the weather.", "</think>",
        #"<tool_call>{"name":"getWeather","arguments":{"location":"Paris"}}</tool_call>"#
      ]
      .enumerated()
      .flatMap { index, chunk in
        parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
      } + parser.finish()

    expectNoDifference(
      parts,
      [
        .reasoning("I need the weather."),
        .toolCall(EdgeRawToolCall(name: "getWeather", arguments: ["location": "Paris"]))
      ]
    )
  }

  @Test
  func `Uses The Raw Response When Parts Are Unavailable`() {
    let generation = EdgeToolsEngineGeneration(
      prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
      decodeMetrics: EdgeToolsDecodeMetrics(
        tokens: 0,
        duration: .zero,
        durationToFirstToken: .zero
      ),
      wasStopped: false,
      tokens: [],
      response: "The forecast is sunny."
    )

    expectNoDifference(
      EdgeToolsLLMPrompt.Message(generation: generation),
      .assistant([.text("The forecast is sunny.")])
    )
  }
}

private func assertReasoningParts<Parser: EdgeToolsGenerationParser>(
  parser: inout Parser,
  prefix: String = "",
  chunks: [String]
) {
  if !prefix.isEmpty {
    _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
  }
  let parts =
    chunks.enumerated()
    .flatMap { index, chunk in
      parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
    } + parser.finish()
  expectNoDifference(parts, [.reasoning("check the weather"), .text("Sunny.")])
}

private func assertTextParts<Parser: EdgeToolsGenerationParser>(
  parser: inout Parser,
  prefix: String,
  chunks: [String]
) {
  _ = parser.accept(token: EdgeToolsToken(id: -1, stringValue: prefix))
  let parts =
    chunks.enumerated()
    .flatMap { index, chunk in
      parser.accept(token: EdgeToolsToken(id: index, stringValue: chunk))
    } + parser.finish()
  expectNoDifference(parts, [.text("Sunny.")])
}
