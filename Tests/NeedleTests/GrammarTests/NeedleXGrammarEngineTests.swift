#if SwiftNeedleXGrammar && SwiftNeedleSentencepiece
  import CustomDump
  import Foundation
  import Needle
  import Testing

  // MARK: - Suite

  @Suite
  struct `NeedleXGrammarEngine tests` {
    private let engine: NeedleXGrammarEngine
    private let tokenizer: NeedleSPTokenizingModel

    init() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
      self.tokenizer = tokenizer
      self.engine = try #require(NeedleXGrammarEngine(tokenizer: tokenizer))
    }

    @Test
    func `Compile Tools With Empty Tools Array`() async {
      await #expect(throws: Never.self) {
        _ = try await self.engine.compile(tools: [])
      }
    }

    @Test
    func `Compile Tools With Single Tool`() async {
      await #expect(throws: Never.self) {
        _ = try await self.engine.compile(tools: [.sendEmail])
      }
    }

    @Test
    func `Compile Tools With Multiple Tools`() async {
      await #expect(throws: Never.self) {
        _ = try await self.engine.compile(tools: [.sendEmail, .getWeather])
      }
    }

    @Test
    func `Compile Tools With Complex Tool`() async {
      await #expect(throws: Never.self) {
        _ = try await self.engine.compile(tools: [.complexTool])
      }
    }

    @Suite
    struct `Range tests` {
      private let tokenizer: NeedleSPTokenizingModel
      private let eosToken: NeedleToken.ID

      init() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.eosToken = try #require(tokenizer.eosTokenId)
      }

      private func makeEngine(
        toolCallInvocationRange: NeedleXGrammarEngine.ToolCallInvocationRange =
          .unbounded(minimum: 0)
      ) throws -> NeedleXGrammarEngine {
        let engine = try #require(NeedleXGrammarEngine(tokenizer: self.tokenizer))
        engine.toolCallInvocationRange = toolCallInvocationRange
        return engine
      }

      @Test
      func `Default Unbounded Range Accepts Empty Tool Call List`() async throws {
        let engine = try self.makeEngine()
        let matcher = try await engine.compile(tools: [.getWeather])
        assertAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Default Unbounded Range Accepts Multiple Tool Calls`() async throws {
        let engine = try self.makeEngine()
        let matcher = try await engine.compile(tools: [.getWeather])
        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#
        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Unbounded With Min One Rejects Empty Tool Call List`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .unbounded(minimum: 1))
        let matcher = try await engine.compile(tools: [.getWeather])
        assertRejects(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Unbounded With Min One Accepts Single Tool Call`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .unbounded(minimum: 1))
        let matcher = try await engine.compile(tools: [.getWeather])
        assertAccepts(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Bounded Max One Rejects Two Tool Calls`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .bounded(0...1))
        let matcher = try await engine.compile(tools: [.getWeather])
        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#
        assertRejects(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Bounded Max One Accepts Empty And Single Tool Call`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .bounded(0...1))
        let emptyMatcher = try await engine.compile(tools: [.getWeather])
        assertAccepts(
          #"<tool_call> []"#,
          matcher: emptyMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        let singleMatcher = try await engine.compile(tools: [.getWeather])
        assertAccepts(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: singleMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Bounded Range Rejects Out Of Range Counts`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .bounded(2...3))
        let singleMatcher = try await engine.compile(tools: [.getWeather])
        assertRejects(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
          matcher: singleMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
        let pairMatcher = try await engine.compile(tools: [.getWeather])
        assertAccepts(
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}}]"#,
          matcher: pairMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Empty Tools Ignores Invocation Range`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .bounded(1...1))
        let matcher = try await engine.compile(tools: [])
        assertAccepts(
          #"<tool_call> []"#,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }

      @Test
      func `Tool Call Invocation Range Is Mutable On Engine`() throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .bounded(2...5))
        expectNoDifference(engine.toolCallInvocationRange, .bounded(2...5))
        engine.toolCallInvocationRange = .bounded(1...2)
        expectNoDifference(engine.toolCallInvocationRange, .bounded(1...2))
      }

      @Test
      func `Exact Three Accepts Three Tool Calls`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .exact(3))
        let matcher = try await engine.compile(tools: [.getWeather])
        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Paris"}},{"name":"get_weather","arguments":{"location":"Tokyo"}}]"#
        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Exact Zero Accepts Only Empty Tool Call List`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .exact(0))
        let matcher = try await engine.compile(tools: [.getWeather])
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
      func `Negative Minimum Tool Calls Throws Error`() async throws {
        let engine = try self.makeEngine(toolCallInvocationRange: .unbounded(minimum: -1))
        await #expect(throws: NeedleXGrammarEngineError.invalidToolInvocationRange) {
          _ = try await engine.compile(tools: [.getWeather])
        }
      }
    }

    @Suite
    struct `Matcher tests` {
      private let engine: NeedleXGrammarEngine
      private let tokenizer: NeedleSPTokenizingModel
      private let eosToken: NeedleToken.ID

      init() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.eosToken = try #require(tokenizer.eosTokenId)
        self.engine = try #require(NeedleXGrammarEngine(tokenizer: tokenizer))
      }

      @Test
      func `Reset Restores Initial State`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])
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
      func `Rollback Allows Accepting Alternative Branch`() async throws {
        let matcher = try await self.engine.compile(tools: [.sendEmail, .getWeather])

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
        expectNoDifference(matcher.accept(tokenId: NeedleToken.ID(firstAllowed)), true)

        matcher.rollback(1)
        expectNoDifference(matcher.isCompleted, false)
      }

      @Test
      func `Completion State Transitions`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])
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
      func `Fork Preserves Accept State`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

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
      func `Bitmask Disallows Eos Before Completion`() async throws {
        let matcher = try await self.engine.compile(tools: [.sendEmail])

        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask[self.eosToken], false)
      }

      @Test
      func `Bitmask Allows Eos After Completion`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])
        let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        for tokenId in encodedGrammarText(call, tokenizer: self.tokenizer) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.isCompleted, true)

        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask[self.eosToken], true)
      }

      @Test
      func `Bitmask Has Expected Size`() async throws {
        let matcher = try await self.engine.compile(tools: [.sendEmail])
        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask.count, self.tokenizer.vocabSize)
      }

      @Test
      func `Accepts Valid Complex Tool Call`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"config":{"flags":[true,false],"threshold":0.75},"count":3.5,"enabled":true,"labels":{"ALPHA":1,"BETA_LABEL":2},"mode":"execute","optional_note":null,"priority":4,"routing":{"region":"us-west"},"tags":["a","b"],"ticket_id":"ABC-12","title":"alpha","tuple_args":["alpha",2,true],"window":3}}]"#

        assertAccepts(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Accepts Valid Simple Tool Call`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

        let call = #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        assertAccepts(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Accepts Multiple Consecutive Tool Calls`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

        let calls =
          #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Henry's Altar"}}]"#

        assertAccepts(calls, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Missing Required Field`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"]}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Wrong Type`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":"3.5","enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Invalid Pattern Properties`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"alpha":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Extra Property`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]},"extra":1}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Malformed JSON`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call =
          #"<tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }

      @Test
      func `Rejects Invalid Tool Name`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

        let call = #"toolcall [{"name":"not_a_real_tool","arguments":{"location":"Seoul"}}]"#

        assertRejects(call, matcher: matcher, tokenizer: self.tokenizer, eosToken: self.eosToken)
      }
    }

    @Suite
    struct `Memory usage tests` {
      private let engine: NeedleXGrammarEngine
      private let tokenizer: NeedleSPTokenizingModel

      init() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.engine = try #require(NeedleXGrammarEngine(tokenizer: tokenizer))
      }

      @Test
      func `Cache Size Is Zero Before First Compile`() {
        expectNoDifference(self.engine.cacheSizeBytes, 0)
        expectNoDifference(self.engine.cacheLimitBytes, 0)
      }

      @Test
      func `Cache Size Becomes Non-Negative After Compile`() async throws {
        _ = try await self.engine.compile(tools: [.getWeather])
        expectNoDifference(self.engine.cacheSizeBytes >= 0, true)
      }

      @Test
      func `Cache Limit Becomes Reportable After Compile`() async throws {
        _ = try await self.engine.compile(tools: [.getWeather])
        expectNoDifference(
          self.engine.cacheLimitBytes == -1 || self.engine.cacheLimitBytes >= 0,
          true
        )
      }

      @Test
      func `Matcher Reports Non-Zero Memory Size`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])
        expectNoDifference(matcher.memorySizeBytes > 0, true)
      }

      @Test
      func `Forked Matcher Reports Equal Memory Size`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])
        let forked = matcher.fork()
        expectNoDifference(forked.memorySizeBytes, matcher.memorySizeBytes)
      }

      @Test
      func `More Tools Yield Larger Compiled Grammar`() async throws {
        let single = try await self.engine.compile(tools: [.getWeather])
        let many = try await self.engine.compile(tools: [.getWeather, .sendEmail, .complexTool])
        expectNoDifference(many.memorySizeBytes > single.memorySizeBytes, true)
      }
    }
  }

  // MARK: - Helpers

  private func firstRejectedToken(
    in text: String,
    matcher: NeedleXGrammarEngine.Matcher,
    tokenizer: NeedleSPTokenizingModel
  ) -> (index: Int, tokenId: NeedleToken.ID, token: String, prefix: String)? {
    let tokenIds = encodedGrammarText(text, tokenizer: tokenizer)
    for (index, tokenId) in tokenIds.enumerated() {
      guard !matcher.accept(tokenId: tokenId) else { continue }
      let token = tokenizer.tokens(from: [tokenId]).first ?? ""
      let prefix = tokenizer.decode(tokenIds: tokenIds.prefix(index + 1))
      return (index, tokenId, token, prefix)
    }
    return nil
  }

  private func encodedGrammarText(
    _ text: String,
    tokenizer: NeedleSPTokenizingModel
  ) -> [NeedleToken.ID] {
    let tokenIds = tokenizer.encode(text: text)
    guard let firstTokenId = tokenIds.first else { return tokenIds }
    let firstToken = tokenizer.tokens(from: [firstTokenId]).first ?? ""
    if firstToken.hasPrefix("▁") {
      return Array(tokenIds.dropFirst())
    }
    return tokenIds
  }

  private func assertAccepts(
    _ text: String,
    matcher: NeedleXGrammarEngine.Matcher,
    tokenizer: NeedleSPTokenizingModel,
    eosToken: NeedleToken.ID
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
    matcher: NeedleXGrammarEngine.Matcher,
    tokenizer: NeedleSPTokenizingModel,
    eosToken: NeedleToken.ID
  ) {
    guard firstRejectedToken(in: text, matcher: matcher, tokenizer: tokenizer) == nil else {
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), false)
  }

#endif
