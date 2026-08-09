import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `LFM2P5 tests` {
  @Suite
  struct `LFM2P5PythonToolCallParser tests` {
    @Test
    func `Decodes Escaped Unicode Surrogate Pairs`() throws {
      var parser = LFM2P5PythonToolCallParser()
      let source = #"<|tool_call_start|>[emoji(value='\uD83D\uDE00')]<|tool_call_end|>"#
      let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
      let call = try #require(parsed.first)

      expectNoDifference(call.arguments, ["value": "😀"])
    }
  }

  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `LFM2P5MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await LFM2P5MLXModelEngine(from: downloadLFM2P5())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }
    }
  #endif

  #if XGrammar
    @Suite(.serialized)
    struct `LFM2P5XGRCompiler tests`: ~Copyable {
      private let compiler: XGRCompiler
      private let tokenizer: NeedleSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.compiler = compiler
        self.eosToken = eosToken
      }

      @Test
      func `LFM2P5 Python Removes Duplicate EBNF Rules`() throws {
        let grammar = try XGRGrammar.lfm2P5Python(
          tools: [.getWeather, Self.getForecast],
          range: .exact(1)
        )
        let expressions = Self.expressions(in: grammar.ebnf)

        expectNoDifference(Set(expressions).count, expressions.count)
      }

      @Test
      func `LFM2P5 Python Uses Keyword Arguments And Nested Dictionaries`() throws {
        let validMatcher = try self.compiler.makeMatcher(
          try XGRGrammar.lfm2P5Python(tools: [Self.nestedTool], range: .exact(1))
        )
        let valid =
          """
          <|tool_call_start|>[nested(payload={"enabled":True,"child":{"missing":None,"items":[\
          {"flag":False}]}})]<|tool_call_end|>
          """
        assertGrammarAccepts(
          valid,
          matcher: validMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let optionalMatcher = try self.compiler.makeMatcher(
          try XGRGrammar.lfm2P5Python(tools: [Self.nestedTool], range: .exact(1))
        )
        assertGrammarAccepts(
          """
          <|tool_call_start|>[nested(payload={"enabled":True,"child":{"missing":None,"items":[]}},\
          note="hello")]<|tool_call_end|>
          """,
          matcher: optionalMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let topLevelJSONMatcher = try self.compiler.makeMatcher(
          try XGRGrammar.lfm2P5Python(tools: [Self.nestedTool], range: .exact(1))
        )
        assertGrammarRejects(
          #"<|tool_call_start|>[nested("payload":{"enabled":True,"child":{"missing":None,"items":[]}})]<|tool_call_end|>"#,
          matcher: topLevelJSONMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let nestedKeywordMatcher = try self.compiler.makeMatcher(
          try XGRGrammar.lfm2P5Python(tools: [Self.nestedTool], range: .exact(1))
        )
        assertGrammarRejects(
          #"<|tool_call_start|>[nested(payload={"enabled"=True,"child":{"missing":None,"items":[]}})]<|tool_call_end|>"#,
          matcher: nestedKeywordMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      private static func expressions(in ebnf: String) -> [String] {
        ebnf.split(separator: "\n")
          .compactMap { line in
            guard let separator = line.range(of: "::=") else { return nil }
            return line[separator.upperBound...].trimmingCharacters(in: .whitespaces)
          }
      }

      private static var getForecast: EdgeToolDefinition {
        var tool = EdgeToolDefinition.getWeather
        tool.name = "getForecast"
        return tool
      }

      private static let nestedTool = EdgeToolDefinition(
        name: "nested",
        description: "Exercises nested Python values.",
        arguments: EdgeToolsGenerationSchema(
          .type(.object),
          .properties([
            "payload": EdgeToolsGenerationSchema(
              .type(.object),
              .properties([
                "enabled": .boolean,
                "child": EdgeToolsGenerationSchema(
                  .type(.object),
                  .properties([
                    "missing": .null,
                    "items": EdgeToolsGenerationSchema(
                      .type(.array),
                      .items(
                        EdgeToolsGenerationSchema(
                          .type(.object),
                          .properties(["flag": .boolean]),
                          .required(["flag"]),
                          .additionalProperties(false)
                        )
                      )
                    )
                  ]),
                  .required(["missing", "items"]),
                  .additionalProperties(false)
                )
              ]),
              .required(["enabled", "child"]),
              .additionalProperties(false)
            ),
            "note": .string
          ]),
          .required(["payload"]),
          .additionalProperties(false)
        )
      )
    }
  #endif
}
