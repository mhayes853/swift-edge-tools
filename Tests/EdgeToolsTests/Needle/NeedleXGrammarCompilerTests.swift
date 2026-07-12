#if XGrammar && Sentencepiece
  import CustomDump
  import Foundation
  import EdgeTools
  import Testing

  // MARK: - Suite

  @Suite
  struct `NeedleXGrammarCompiler tests` {
    private let engine: XGrammarCompiler
    private let tokenizer: NeedleSPTokenizer

    init() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      self.tokenizer = tokenizer
      self.engine = try #require(XGrammarCompiler.needle(tokenizer: tokenizer))
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
    struct `Range tests` {
      private let tokenizer: NeedleSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.eosToken = try #require(tokenizer.eosTokenId)
      }

      private func makeEngine() throws -> XGrammarCompiler {
        try #require(XGrammarCompiler.needle(tokenizer: self.tokenizer))
      }

      @Test
      func `Default Unbounded Range Accepts Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        assertAccepts(
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
        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Unbounded With Min One Rejects Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .unbounded(minimum: 1))
        )
        assertRejects(
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
        assertAccepts(
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
        assertRejects(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Bounded Max One Accepts Empty And Single Tool Call`() throws {
        let engine = try self.makeEngine()
        let emptyMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
        )
        assertAccepts(
          #"<tool_call> []"#,
          matcher: emptyMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        let singleMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(0...1))
        )
        assertAccepts(
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
        assertRejects(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: singleMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let pairMatcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .bounded(2...3))
        )
        assertAccepts(
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
        assertAccepts(
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
        assertAccepts(
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
        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Exact Zero Accepts Only Empty Tool Call List`() throws {
        let engine = try self.makeEngine()
        let matcher = try engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather], range: .exact(0))
        )
        assertAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        assertRejects(
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
    struct `Matcher tests` {
      private let engine: XGrammarCompiler
      private let tokenizer: NeedleSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.eosToken = try #require(tokenizer.eosTokenId)
        self.engine = try #require(XGrammarCompiler.needle(tokenizer: tokenizer))
      }

      @Test
      func `Reset Restores Initial State`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        for tokenId in encodedGrammarText(call, tokenizer: self.tokenizer) {
          matcher.accept(tokenId: tokenId)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
        expectNoDifference(matcher.isCompleted, true)
        expectNoDifference(matcher.isTerminated, true)

        matcher.reset()
        expectNoDifference(matcher.isCompleted, false)
        expectNoDifference(matcher.isTerminated, false)
      }

      @Test
      func `Rollback Allows Accepting Alternative Branch`() throws {
        let matcher = try self.engine.compile(
          try XGrammarGrammar.needle(tools: [.sendEmail, .getWeather])
        )

        let firstBitmask = matcher.bitmask()
        let firstAllowedIndex = firstBitmask.storage
          .enumerated()
          .flatMap { word in (0..<32).map { (word.offset, $0) } }
          .first { (wordIndex, bit) in
            (firstBitmask.storage[wordIndex] & (1 << bit)) != 0
          }
        guard let (wordIndex, bit) = firstAllowedIndex else {
          Issue.record("Expected at least one allowed token in the initial bitmask")
          return
        }
        let firstAllowed = wordIndex * 32 + bit
        expectNoDifference(matcher.accept(tokenId: EdgeToolsToken.ID(firstAllowed)), true)

        matcher.rollback(1)
        expectNoDifference(matcher.isCompleted, false)
      }

      @Test
      func `Completion State Transitions`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        expectNoDifference(matcher.isCompleted, false)
        expectNoDifference(matcher.isTerminated, false)

        for tokenId in encodedGrammarText(call, tokenizer: self.tokenizer) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.isCompleted, true)
        expectNoDifference(matcher.isTerminated, false)

        expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
        expectNoDifference(matcher.isCompleted, true)
        expectNoDifference(matcher.isTerminated, true)
      }

      @Test
      func `Fork Preserves Accept State`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

        let tokens = encodedGrammarText(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          tokenizer: self.tokenizer
        )
        let forkPoint = tokens.count / 2
        guard !tokens.isEmpty, forkPoint > 0, forkPoint < tokens.count else {
          Issue.record("Tokenizer produced an unexpected token split for test text")
          return
        }

        for i in 0..<forkPoint {
          expectNoDifference(matcher.accept(tokenId: tokens[i]), true)
        }
        let forked = matcher.fork()

        for i in forkPoint..<tokens.count {
          expectNoDifference(matcher.accept(tokenId: tokens[i]), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
        expectNoDifference(matcher.isTerminated, true)

        for i in forkPoint..<tokens.count {
          expectNoDifference(forked.accept(tokenId: tokens[i]), true)
        }
        expectNoDifference(forked.accept(tokenId: self.eosToken), true)
        expectNoDifference(forked.isTerminated, true)
      }

      @Test
      func `Bitmask Disallows Eos Before Completion`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.sendEmail]))

        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask[self.eosToken], false)
      }

      @Test
      func `Bitmask Allows Eos After Completion`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        for tokenId in encodedGrammarText(call, tokenizer: self.tokenizer) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.isCompleted, true)

        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask[self.eosToken], true)
      }

      @Test
      func `Bitmask Has Expected Size`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.sendEmail]))
        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask.count, 8192)
      }

      @Test
      func `Accepts Valid Complex Tool Call`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1,"BETA_LABEL":2},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#

        assertAccepts(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Accepts Valid Simple Tool Call`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

        let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        assertAccepts(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Accepts Multiple Consecutive Tool Calls`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Henry's Altar"}}]"#

        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Accepts Multiple Distinct Tool Calls`() throws {
        let matcher = try self.engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather, .sendEmail])
        )

        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"send_email","arguments":{"address":"blob@gmail.com","subject":"Hello","body":"World"}}]"#

        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Missing Required Field`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"]}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Wrong Type`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":"3.5","enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Invalid Pattern Properties`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"alpha":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Extra Property`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]},"extra":1}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Malformed JSON`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.complexTool]))

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Invalid Tool Name`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))

        let call = #"toolcall [{"name":"not_a_real_tool","arguments":{"location":"Seoul"}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }
    }

    @Suite
    struct `Memory usage tests` {
      private let engine: XGrammarCompiler
      private let tokenizer: NeedleSPTokenizer

      init() throws {
        let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.engine = try #require(XGrammarCompiler.needle(tokenizer: tokenizer))
      }

      @Test
      func `Cache Size Is Zero Before First Compile`() {
        expectNoDifference(self.engine.cacheSizeBytes, 0)
        expectNoDifference(self.engine.cacheLimitBytes, 0)
      }

      @Test
      func `Cache Size Becomes Non-Negative After Compile`() throws {
        _ = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        expectNoDifference(self.engine.cacheSizeBytes >= 0, true)
      }

      @Test
      func `Cache Limit Becomes Reportable After Compile`() throws {
        _ = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        expectNoDifference(
          self.engine.cacheLimitBytes == -1 || self.engine.cacheLimitBytes >= 0,
          true
        )
      }

      @Test
      func `Matcher Reports Non-Zero Memory Size`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        expectNoDifference(matcher.memorySizeBytes > 0, true)
      }

      @Test
      func `Forked Matcher Reports Equal Memory Size`() throws {
        let matcher = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        let forked = matcher.fork()
        expectNoDifference(forked.memorySizeBytes, matcher.memorySizeBytes)
      }

      @Test
      func `More Tools Yield Larger Compiled Grammar`() throws {
        let single = try self.engine.compile(try XGrammarGrammar.needle(tools: [.getWeather]))
        let many = try self.engine.compile(
          try XGrammarGrammar.needle(tools: [.getWeather, .sendEmail, .complexTool])
        )
        expectNoDifference(many.memorySizeBytes > single.memorySizeBytes, true)
      }
    }
  }

  // MARK: - Helpers

  private func firstRejectedToken(
    in text: String,
    matcher: XGrammarMatcher,
    tokenizer: NeedleSPTokenizer
  ) -> (index: Int, tokenId: EdgeToolsToken.ID, token: String, prefix: String)? {
    let tokenIds = encodedGrammarText(text, tokenizer: tokenizer)
    for (index, tokenId) in tokenIds.enumerated() {
      guard !matcher.accept(tokenId: tokenId) else { continue }
      let token = tokenizer.convertIdToToken(tokenId) ?? ""
      let prefix = tokenizer.decode(tokens: Array(tokenIds.prefix(index + 1)))
      return (index, tokenId, token, prefix)
    }
    return nil
  }

  private func encodedGrammarText(
    _ text: String,
    tokenizer: NeedleSPTokenizer
  ) -> [EdgeToolsToken.ID] {
    let tokenIds = tokenizer.encode(text: text)
    guard let firstTokenId = tokenIds.first else { return tokenIds }
    let firstToken = tokenizer.convertIdToToken(firstTokenId) ?? ""
    if firstToken.hasPrefix("▁") {
      return Array(tokenIds.dropFirst())
    }
    return tokenIds
  }

  private func assertAccepts(
    _ text: String,
    matcher: XGrammarMatcher,
    tokenizer: NeedleSPTokenizer,
    eosToken: EdgeToolsToken.ID
  ) {
    if let rejected = firstRejectedToken(in: text, matcher: matcher, tokenizer: tokenizer) {
      Issue.record(
        "Rejected token \(rejected.tokenId) '\(rejected.token)' at index \(rejected.index) for prefix: \(rejected.prefix)"
      )
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), true)
  }

  private func assertRejects(
    _ text: String,
    matcher: XGrammarMatcher,
    tokenizer: NeedleSPTokenizer,
    eosToken: EdgeToolsToken.ID
  ) {
    guard firstRejectedToken(in: text, matcher: matcher, tokenizer: tokenizer) == nil else {
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), false)
  }

#endif
