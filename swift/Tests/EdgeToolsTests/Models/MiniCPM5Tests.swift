import CustomDump
import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `MiniCPM5 tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `MiniCPM5MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await MiniCPM5MLXModelEngine(from: downloadMiniCPM5())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }
    }
  #endif

  @Suite
  struct `MiniCPM5ToolCallParser tests` {
    @Test
    func `Parses A Tool Call Without Arguments`() throws {
      var parser = MiniCPM5ToolCallParser()
      let source = #"<function name="empty"></function>"#
      let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
      let call = try #require(parsed.first)

      expectNoDifference(call, EdgeRawToolCall(name: "empty", arguments: [:]))
    }

    @Test
    func `Distinguishes Strings From JSON Primitives`() throws {
      var parser = MiniCPM5ToolCallParser()
      let source = #"""
        <function name="values"><param name="string_boolean">"true"</param>
        <param name="boolean">true</param><param name="string_integer">"123"</param>
        <param name="integer">123</param><param name="string_null">"null"</param>
        <param name="null">null</param><param name="cdata"><![CDATA[true]]></param></function>
        """#
      let parsed = parser.accept(token: EdgeToolsToken(id: 0, stringValue: source))
      let call = try #require(parsed.first)

      expectNoDifference(
        call.arguments,
        [
          "string_boolean": "true",
          "boolean": true,
          "string_integer": "123",
          "integer": 123,
          "string_null": "null",
          "null": .null,
          "cdata": "true"
        ]
      )
    }

    @Test
    func `Preserves Tool Markup Inside CDATA Across Every Token Split`() throws {
      let source = #"""
        <function name="record"><param name="text"><![CDATA[first line
        literal </function> and </param> and <function text
        漢字 👩🏽‍💻]]></param></function>
        """#
      let expected = """
        first line
        literal </function> and </param> and <function text
        漢字 👩🏽‍💻
        """

      for splitIndex in source.indices.dropFirst() {
        var parser = MiniCPM5ToolCallParser()
        let first = String(source[..<splitIndex])
        let second = String(source[splitIndex...])
        let firstCall = parser.accept(token: EdgeToolsToken(id: 0, stringValue: first))
        let secondCall = parser.accept(token: EdgeToolsToken(id: 1, stringValue: second))
        let call = try #require((firstCall + secondCall).first)

        expectNoDifference(call.arguments, ["text": .string(expected)])
      }
    }
  }

  #if XGrammar
    @Suite
    struct `MiniCPM5Grammar tests` {
      @Test
      func `Accepts CDATA For String Arguments`() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let matcher = try compiler.makeMatcher(
          try XGRGrammar.miniCPM5(tools: [.getWeather], range: .exact(1))
        )

        assertGrammarAccepts(
          #"""
          <function name="getWeather"><param name="location"><![CDATA[Seoul
          <&]]></param></function>
          """#,
          matcher: matcher,
          tokenizer: tokenizer,
          eosToken: eosToken
        )

        let integerMatcher = try compiler.makeMatcher(
          try XGRGrammar.miniCPM5(tools: [.integerTool], range: .exact(1))
        )
        assertGrammarRejects(
          #"<function name="integerTool"><param name="value"><![CDATA[1]]></param></function>"#,
          matcher: integerMatcher,
          tokenizer: tokenizer,
          eosToken: eosToken
        )
      }

      @Test
      func `Distinguishes Quoted Strings From Typed Primitives`() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let stringGrammar = try XGRGrammar.miniCPM5(
          tools: [.getWeather],
          range: .exact(1)
        )
        let quotedString =
          #"<function name="getWeather"><param name="location">"true"</param></function>"#
        let boolean =
          #"<function name="getWeather"><param name="location">true</param></function>"#

        assertGrammarAccepts(
          quotedString,
          matcher: try compiler.makeMatcher(stringGrammar),
          tokenizer: tokenizer,
          eosToken: eosToken
        )
        assertGrammarRejects(
          boolean,
          matcher: try compiler.makeMatcher(stringGrammar),
          tokenizer: tokenizer,
          eosToken: eosToken
        )

        let booleanTool = EdgeToolDefinition(
          name: "booleanTool",
          description: "Accepts a Boolean.",
          arguments: EdgeToolsGenerationSchema(
            .type(.object),
            .properties(["value": .boolean]),
            .required(["value"]),
            .additionalProperties(false)
          )
        )
        let booleanGrammar = try XGRGrammar.miniCPM5(
          tools: [booleanTool],
          range: .exact(1)
        )
        assertGrammarAccepts(
          #"<function name="booleanTool"><param name="value">true</param></function>"#,
          matcher: try compiler.makeMatcher(booleanGrammar),
          tokenizer: tokenizer,
          eosToken: eosToken
        )
        assertGrammarRejects(
          #"<function name="booleanTool"><param name="value">"true"</param></function>"#,
          matcher: try compiler.makeMatcher(booleanGrammar),
          tokenizer: tokenizer,
          eosToken: eosToken
        )
      }
    }
  #endif
}
