import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct `Needle tests` {
  @Suite
  struct `NeedleModelConfiguration tests` {
    @Test
    func `Decodes Exported Configuration With Optional Defaults`() throws {
      let json =
        #"{"vocab_size":16,"d_model":8,"hidden_size":8,"num_attention_heads":2,"num_kv_heads":1,"num_encoder_layers":1,"num_decoder_layers":1,"num_hidden_layers":1,"max_seq_len":4,"pad_token_id":0,"decoder_start_token_id":1,"tie_word_embeddings":true,"torch_dtype":"float32","decoder_max_length":4}"#

      let configuration =
        try JSONDecoder().decode(
          NeedleModelConfiguration.self,
          from: Data(json.utf8)
        )

      expectNoDifference(configuration.ropeTheta, 10_000)
      expectNoDifference(configuration.rmsNormEps, 1e-6)
      expectNoDifference(configuration.decoderMaxLength, 4)
      expectNoDifference(configuration.dtype, "float32")
    }
  }

  @Suite
  struct `NeedlePrompt tests` {
    @Test
    func `Formats Properly`() throws {
      let prompt = NeedlePrompt(
        system: "You are a helpful assistant who can send emails.",
        user: "Send an email to Henry."
      )

      expectNoDifference(
        try prompt.formatted(tools: [.sendEmail]),
        """
        You are a helpful assistant who can send emails.

        Send an email to Henry.<tools>[{"name":"send_email","description":"Sends an email to a recipient with an email address.",\
        "arguments":{"type":"object","properties":{"address":{"type":"string","description":"The recipient's email address.",\
        "pattern":"[a-z][a-z0-9]{1,10}@gmail\\\\.com","examples":["blob@gmail.com"]},"subject":{"type":"string"},\
        "body":{"type":"string"}},"required":["address","subject","body"]}}]
        """
      )
    }

    @Test
    func `Uses Canonical Tool And Schema Field Order`() throws {
      let prompt = NeedlePrompt(system: "", user: "Weather?")

      let rendered = try prompt.formatted(tools: [.getWeather])
      let jsonStart = rendered.firstIndex(of: "[") ?? rendered.endIndex
      let jsonSlice = rendered[jsonStart...]

      let expected =
        """
        [{"name":"get_weather","description":"Gets the current weather for a location.",\
        "arguments":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"],\
        "additionalProperties":false}}]
        """
      expectNoDifference(String(jsonSlice), expected)
    }

    @Test(
      arguments: [
        ("sendEmail", "send_email"),
        ("sendEmailTo", "send_email_to"),
        ("SendEmailTo", "send_email_to"),
        ("send_email", "send_email"),
        ("", ""),
        ("sendEmail2", "send_email2"),
        ("send", "send"),
        ("Send", "send")
      ]
    )
    func `Formats Normalized Tool Names`(name: String, expectedName: String) throws {
      let tool = EdgeToolDefinition(
        name: name,
        description: "Blob",
        arguments: EdgeToolsGenerationSchema(
          .type(.object),
          .properties(["name": .string])
        ),
        includesSchemaInInstructions: true
      )

      let prompt = NeedlePrompt(system: "", user: "")
      let formatted = try prompt.formatted(tools: [tool])

      expectNoDifference(formatted.contains("\"name\":\"\(expectedName)\""), true)
    }

    @Test
    func `Omits Schemas Known By The Model`() throws {
      let prompt = NeedlePrompt(system: "", user: "")
      var innate = EdgeToolDefinition.sendEmail
      innate.includesSchemaInInstructions = false

      let formatted = try prompt.formatted(tools: [innate, .getWeather])

      expectNoDifference(formatted.contains(#""name":"send_email""#), false)
      expectNoDifference(formatted.contains(#""name":"get_weather""#), true)
    }

    @Test
    func `Formatting Escapes Quotes`() throws {
      let tool = EdgeToolDefinition(
        name: "say_\"hello\"",
        description: "Uses a \"quoted\" phrase",
        arguments: EdgeToolsGenerationSchema(
          .type(.object),
          .description("Schema with \"quotes\""),
          .properties([
            "message": EdgeToolsGenerationSchema(
              .string,
              .pattern(#"say \"hi\""#)
            )
          ])
        ),
        includesSchemaInInstructions: true
      )

      let prompt = NeedlePrompt(system: "", user: "")
      let formatted = try prompt.formatted(tools: [tool])

      expectNoDifference(formatted.contains(#""name":"say_\"hello\""#), true)
      expectNoDifference(formatted.contains(#""description":"Uses a \"quoted\" phrase"#), true)
      expectNoDifference(formatted.contains(#""pattern":"say \\\"hi\\\""#), true)
    }
  }

  @Suite
  struct `NeedleToolCallParser tests` {
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
        calls.append(contentsOf: parser.accept(token: token))
      }
      return calls
    }

  }

}

#if XGrammar
  @Suite
  struct `NeedleXGRCompiler tests`: ~Copyable {
    @Suite
    struct `Range tests`: ~Copyable {
      private let tokenizer: NeedleSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try testTokenizer()
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.eosToken = eosToken
      }

      private func makeEngine() throws -> XGRCompiler {
        try requiredNeedleCompiler(tokenizer: self.tokenizer)
      }

      @Test
      func `Default Unbounded Range Accepts Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(try XGRGrammar.needle(tools: [.getWeather]))
        assertGrammarAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Default Unbounded Range Accepts Multiple Tool Calls`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(try XGRGrammar.needle(tools: [.getWeather]))
        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#
        assertGrammarAccepts(
          calls,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Unbounded With Min One Rejects Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: 1))
        )
        assertGrammarRejects(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Unbounded With Min One Accepts Single Tool Call`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: 1))
        )
        assertGrammarAccepts(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Bounded Max One Rejects Two Tool Calls`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
        )
        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#
        assertGrammarRejects(
          calls,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Bounded Max One Accepts Empty And Single Tool Call`() throws {
        let engine = try self.makeEngine()
        let emptyMatcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
        )
        assertGrammarAccepts(
          #"<tool_call> []"#,
          matcher: emptyMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        let singleMatcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
        )
        assertGrammarAccepts(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: singleMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Bounded Range Rejects Out Of Range Counts`() throws {
        let engine = try self.makeEngine()
        let singleMatcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .bounded(2...3))
        )
        assertGrammarRejects(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: singleMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let pairMatcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .bounded(2...3))
        )
        assertGrammarAccepts(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#,
          matcher: pairMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Empty Tools Ignores Invocation Range`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [], range: .bounded(1...1))
        )
        assertGrammarAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Exact Three Accepts Three Tool Calls`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .exact(3))
        )
        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}},{"name":"get_weather","arguments":{"location":"Tokyo"}}]"#
        assertGrammarAccepts(
          calls,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Exact Zero Accepts Only Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.makeMatcher(
          try XGRGrammar.needle(tools: [.getWeather], range: .exact(0))
        )
        assertGrammarAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        assertGrammarRejects(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Negative Minimum Tool Calls Throws Error`() throws {
        let engine = try self.makeEngine()
        let error = #expect(throws: XGRError.self) {
          _ = try engine.makeMatcher(
            try XGRGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: -1))
          )
        }
        expectNoDifference(error?.code, XGRError.Code.invalidToolInvocationRange)
      }
    }
  }
#endif

#if XGrammar
  @Suite
  struct `NeedleXGRMatcher tests`: ~Copyable {
    private let engine: XGRCompiler
    private let tokenizer: NeedleSPTokenizer
    private let eosToken: EdgeToolsToken.ID

    init() throws {
      let tokenizer = try testTokenizer()
      let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
      let engine = try requiredNeedleCompiler(tokenizer: tokenizer)
      self.tokenizer = tokenizer
      self.eosToken = eosToken
      self.engine = engine
    }
    @Test
    func `Accepts Valid Complex Tool Call`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.complexTool]))

      let call = #"""
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},
        "labels":{"ALPHA":1,"BETA_LABEL":2},"window":3,"tuple_args":["alpha",2,true],
        "optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
        """#
        .replacingOccurrences(of: "\n", with: "")

      assertGrammarAccepts(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Accepts Valid Simple Tool Call`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.getWeather]))

      let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

      assertGrammarAccepts(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Accepts Multiple Consecutive Tool Calls`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.getWeather]))

      let calls =
        #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Henry's Altar"}}]"#

      assertGrammarAccepts(
        calls,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Accepts Multiple Distinct Tool Calls`() throws {
      let matcher = try self.engine.makeMatcher(
        try XGRGrammar.needle(tools: [.getWeather, .sendEmail])
      )

      let calls =
        #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"send_email","arguments":{"address":"blob@gmail.com","subject":"Hello","body":"World"}}]"#

      assertGrammarAccepts(
        calls,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Rejects Missing Required Field`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.complexTool]))

      let call = #"""
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},
        "labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,
        "tags":["a","b"]}}]
        """#
        .replacingOccurrences(of: "\n", with: "")

      assertGrammarRejects(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Rejects Wrong Type`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.complexTool]))

      let call = #"""
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":"3.5","enabled":true,
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},
        "labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,
        "tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
        """#
        .replacingOccurrences(of: "\n", with: "")

      assertGrammarRejects(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Rejects Invalid Pattern Properties`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.complexTool]))

      let call = #"""
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},
        "labels":{"alpha":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,
        "tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
        """#
        .replacingOccurrences(of: "\n", with: "")

      assertGrammarRejects(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Rejects Extra Property`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.complexTool]))

      let call = #"""
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},
        "labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,
        "tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]},"extra":1}}]
        """#
        .replacingOccurrences(of: "\n", with: "")

      assertGrammarRejects(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Rejects Malformed JSON`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.complexTool]))

      let call = #"""
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},
        "labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,
        "tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}]
        """#
        .replacingOccurrences(of: "\n", with: "")

      assertGrammarRejects(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    @Test
    func `Rejects Invalid Tool Name`() throws {
      let matcher = try self.engine.makeMatcher(try XGRGrammar.needle(tools: [.getWeather]))

      let call = #"toolcall [{"name":"not_a_real_tool","arguments":{"location":"Seoul"}}]"#

      assertGrammarRejects(
        call,
        matcher: matcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }
  }
#endif

#if XGrammar
  // MARK: - Helpers

  private func requiredNeedleCompiler(
    tokenizer: some XGRTokenizer
  ) throws -> XGRCompiler {
    let tokenizerInfo = try XGRTokenizerInfo.needle(tokenizer: tokenizer)
    return try XGRCompiler(tokenizerInfo: tokenizerInfo)
  }

  private func requiredNeedleCompiler(
    erasedTokenizer tokenizer: any XGRTokenizer
  ) throws -> XGRCompiler {
    let tokenizerInfo = try XGRTokenizerInfo.needle(tokenizer: tokenizer)
    return try XGRCompiler(tokenizerInfo: tokenizerInfo)
  }
#endif
