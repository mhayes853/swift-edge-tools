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
