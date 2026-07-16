#if XGrammar && Sentencepiece
  import CustomDump
  import Foundation
  import EdgeTools
  import Testing

  // MARK: - Suite

  @Suite
  struct `NeedleXGrammarCompiler tests`: ~Copyable {
    private let engine: XGrammarCompiler
    private let tokenizer: EdgeToolsSPTokenizer

    init() throws {
      let tokenizer = try makeTestTokenizer()
      let engine = try requiredNeedleCompiler(tokenizer: tokenizer)
      self.tokenizer = tokenizer
      self.engine = engine
    }

    @Test
    func `Compile Tools With Empty Tools Array`() {
      #expect(throws: Never.self) {
        _ = try self.engine.compile(try XGrammarGrammar.needle(tools: []))
      }
    }

    @Test
    func `Compile Tools With Single Tool`() {
      #expect(throws: Never.self) {
        _ = try self.engine.compile(try XGrammarGrammar.needle(tools: [.sendEmail]))
      }
    }

    @Test
    func `Compile Tools With Multiple Tools`() {
      #expect(throws: Never.self) {
        _ = try self.engine.compile(try XGrammarGrammar.needle(tools: [.sendEmail, .getWeather]))
      }
    }

    @Test
    func `Compile Tools With Complex Tool`() {
      #expect(throws: Never.self) {
        _ = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))
      }
    }

    @Suite
    struct `Range tests`: ~Copyable {
      private let tokenizer: EdgeToolsSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try makeTestTokenizer()
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.eosToken = eosToken
      }

      private func makeEngine() throws -> XGrammarCompiler {
        try requiredNeedleCompiler(tokenizer: self.tokenizer)
      }

      @Test
      func `Default Unbounded Range Accepts Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
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
        let matcher = try engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
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
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: 1))
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
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: 1))
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
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
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
        let emptyMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
        )
        assertGrammarAccepts(
          #"<tool_call> []"#,
          matcher: emptyMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        let singleMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
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
        let singleMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(2...3))
        )
        assertGrammarRejects(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: singleMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let pairMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(2...3))
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
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [], range: .bounded(1...1))
        )
        assertGrammarAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Compile Accepts Explicit Tool Call Invocation Range`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .exact(0))
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
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .exact(3))
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
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .exact(0))
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
        #expect(throws: NeedleXGrammarError.invalidToolInvocationRange) {
          _ = try engine.compile(
            try XGrammarGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: -1))
          )
        }
      }
    }

    @Suite
    struct `Matcher tests`: ~Copyable {
      private let engine: XGrammarCompiler
      private let tokenizer: EdgeToolsSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try makeTestTokenizer()
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let engine = try requiredNeedleCompiler(tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.eosToken = eosToken
        self.engine = engine
      }
      @Test
      func `Accepts Valid Complex Tool Call`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          [
            #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id""#,
            #":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1,"BETA_LABEL":2},"window":3,"tuple_args":["alph"#,
            #"a",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#
          ]
          .joined()

        assertGrammarAccepts(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Accepts Valid Simple Tool Call`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

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
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

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
        let matcher = try self.engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather, .sendEmail])
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
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          [
            #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id""#,
            #":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"opt"#,
            #"ional_note":null,"tags":["a","b"]}}]"#
          ]
          .joined()

        assertGrammarRejects(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Rejects Wrong Type`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          [
            #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":"3.5","enabled":true,"mode":"execute","ticket_i"#,
            #"d":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"o"#,
            #"ptional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#
          ]
          .joined()

        assertGrammarRejects(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Rejects Invalid Pattern Properties`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          [
            #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id""#,
            #":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"alpha":1},"window":3,"tuple_args":["alpha",2,true],"opt"#,
            #"ional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#
          ]
          .joined()

        assertGrammarRejects(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Rejects Extra Property`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          [
            #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id""#,
            #":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"opt"#,
            #"ional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]},"extra":1}}]"#
          ]
          .joined()

        assertGrammarRejects(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Rejects Malformed JSON`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          [
            #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id""#,
            #":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"opt"#,
            #"ional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}]"#
          ]
          .joined()

        assertGrammarRejects(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Rejects Invalid Tool Name`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

        let call = #"toolcall [{"name":"not_a_real_tool","arguments":{"location":"Seoul"}}]"#

        assertGrammarRejects(
          call,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }

  }

  // MARK: - Helpers

  private func requiredNeedleCompiler(
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) throws -> XGrammarCompiler {
    guard let compiler = XGrammarCompiler.needle(tokenizer: tokenizer) else {
      throw XGrammarError(message: "Needle requires a tokenizer with an EOS token.")
    }
    return compiler
  }

  private func requiredNeedleCompiler(
    erasedTokenizer tokenizer: borrowing any EdgeToolsTokenizer & ~Copyable
  ) throws -> XGrammarCompiler {
    guard let compiler = XGrammarCompiler.needle(tokenizer: tokenizer) else {
      throw XGrammarError(message: "Needle requires a tokenizer with an EOS token.")
    }
    return compiler
  }
#endif
