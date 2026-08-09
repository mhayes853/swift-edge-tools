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
      func `Requires CDATA For String Arguments That Spell A JSON Primitive`() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let grammar = try XGRGrammar.miniCPM5(tools: [.getWeather], range: .exact(1))
        let call = { (value: String) in
          #"<function name="getWeather"><param name="location">\#(value)</param></function>"#
        }

        // NB: A raw value that parses back as a primitive would decode to the wrong type, and a
        // trailing run of spaces is trimmed away before parsing, so both have to be unreachable.
        for value in [
          "true", "false", "null", "123", "-4", "1.5", #""Seoul""#, "{}", "[]", "true "
        ] {
          assertGrammarRejects(
            call(value),
            matcher: try compiler.makeMatcher(grammar),
            tokenizer: tokenizer,
            eosToken: eosToken
          )
        }
        for value in ["Seoul", "tuesday", "farewell", "nullify", "true story", "<![CDATA[true]]>"] {
          assertGrammarAccepts(
            call(value),
            matcher: try compiler.makeMatcher(grammar),
            tokenizer: tokenizer,
            eosToken: eosToken
          )
        }
      }

      @Test
      func `Emits Raw Values For String Arguments Held In Their Own Rules`() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let tool = EdgeToolDefinition(
          name: "shapedTool",
          description: "Accepts strings whose schemas compile to their own rules.",
          arguments: EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "mode": EdgeToolsGenerationSchema(.string, .enum([.string("execute")])),
              "ticket": EdgeToolsGenerationSchema(.string, .pattern("[A-Z]{3}-[0-9]{2}")),
              "priority": EdgeToolsGenerationSchema(.type([.string, .integer])),
              "note": EdgeToolsGenerationSchema(.type([.string, .null]))
            ]),
            .required(["mode", "ticket", "priority", "note"]),
            .additionalProperties(false)
          )
        )
        let grammar = try XGRGrammar.miniCPM5(tools: [tool], range: .exact(1))
        let call = { (mode: String, ticket: String, priority: String, note: String) in
          """
          <function name="shapedTool"><param name="mode">\(mode)</param>\
          <param name="ticket">\(ticket)</param><param name="priority">\(priority)</param>\
          <param name="note">\(note)</param></function>
          """
        }

        for arguments in [
          ("execute", "ABC-12", "4", "null"),
          ("execute", "ABC-12", "urgent", "look into it")
        ] {
          assertGrammarAccepts(
            call(arguments.0, arguments.1, arguments.2, arguments.3),
            matcher: try compiler.makeMatcher(grammar),
            tokenizer: tokenizer,
            eosToken: eosToken
          )
        }
        for arguments in [
          (#""execute""#, "ABC-12", "4", "null"),
          ("execute", #""ABC-12""#, "4", "null"),
          ("execute", "ABC-12", #""urgent""#, "null"),
          ("execute", "ABC-12", "4", #""look into it""#)
        ] {
          assertGrammarRejects(
            call(arguments.0, arguments.1, arguments.2, arguments.3),
            matcher: try compiler.makeMatcher(grammar),
            tokenizer: tokenizer,
            eosToken: eosToken
          )
        }
      }

      @Test
      func `Keeps Strings Nested Inside Container Arguments Quoted`() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let tool = EdgeToolDefinition(
          name: "routingTool",
          description: "Accepts an object argument.",
          arguments: EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "routing": EdgeToolsGenerationSchema(
                .type(.object),
                .properties(["region": .string]),
                .required(["region"]),
                .additionalProperties(false)
              )
            ]),
            .required(["routing"]),
            .additionalProperties(false)
          )
        )
        let grammar = try XGRGrammar.miniCPM5(tools: [tool], range: .exact(1))
        let call = { (value: String) in
          #"<function name="routingTool"><param name="routing">\#(value)</param></function>"#
        }

        assertGrammarAccepts(
          call(#"{"region":"us-west"}"#),
          matcher: try compiler.makeMatcher(grammar),
          tokenizer: tokenizer,
          eosToken: eosToken
        )
        assertGrammarRejects(
          call("{region:us-west}"),
          matcher: try compiler.makeMatcher(grammar),
          tokenizer: tokenizer,
          eosToken: eosToken
        )
      }
    }
  #endif
}
