import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `MiniCPM5 tests` {
  #if MLX && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `MiniCPM5MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await MiniCPM5MLXModelEngine(from: downloadMiniCPM5())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Generates Reasoning Snapshot`() async throws {
        let engine = try await MiniCPM5MLXModelEngine(from: downloadMiniCPM5())
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }
    }
  #endif

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

  #if HuggingFaceTokenizers && Llama && canImport(CLlama) && !os(WASI)
    @Suite(.serialized)
    struct `MiniCPM5LlamaModelEngine tests` {
      @Test
      func `Llama Completes Tool Turn Snapshot`() async throws {
        let engine = try MiniCPM5LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .miniCPM5)).path()
        )
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Llama Generates Reasoning Snapshot`() async throws {
        let engine = try MiniCPM5LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .miniCPM5)).path()
        )
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }
    }
  #endif
}
