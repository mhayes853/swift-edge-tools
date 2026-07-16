#if XGrammar && Sentencepiece
  import CustomDump
  import EdgeTools
  import Testing

  private let genericGrammarText =
    #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

  @Suite
  struct `XGrammarMatcher tests`: ~Copyable {
    private let engine: XGrammarCompiler
    private let tokenizer: EdgeToolsSPTokenizer
    private let eosToken: EdgeToolsToken.ID

    init() throws {
      let tokenizer = try makeTestTokenizer()
      let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
      let engine = try makeGenericXGrammarCompiler(tokenizer: tokenizer)
      self.tokenizer = tokenizer
      self.eosToken = eosToken
      self.engine = engine
    }

    @Test
    func `Reset Restores Initial State`() throws {
      let matcher = try self.engine.compile(try genericGrammar())
      let call = genericGrammarText

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
      let matcher = try self.engine.compile(try genericGrammar())

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
      let matcher = try self.engine.compile(try genericGrammar())
      let call = genericGrammarText

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
      let matcher = try self.engine.compile(try genericGrammar())

      let tokens = encodedGrammarText(
        genericGrammarText,
        tokenizer: self.tokenizer
      )
      let forkPoint = tokens.count / 2
      guard !tokens.isEmpty, forkPoint > 0, forkPoint < tokens.count else {
        Issue.record("Tokenizer produced an unexpected token split for test text")
        return
      }

      for tokenIndex in 0..<forkPoint {
        expectNoDifference(matcher.accept(tokenId: tokens[tokenIndex]), true)
      }
      let forked = try matcher.fork()

      for tokenIndex in forkPoint..<tokens.count {
        expectNoDifference(matcher.accept(tokenId: tokens[tokenIndex]), true)
      }
      expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
      expectNoDifference(matcher.isTerminated, true)

      for tokenIndex in forkPoint..<tokens.count {
        expectNoDifference(forked.accept(tokenId: tokens[tokenIndex]), true)
      }
      expectNoDifference(forked.accept(tokenId: self.eosToken), true)
      expectNoDifference(forked.isTerminated, true)
    }

    @Test
    func `Bitmask Disallows Eos Before Completion`() throws {
      let matcher = try self.engine.compile(try genericGrammar())

      let bitmask = matcher.bitmask()
      expectNoDifference(bitmask[self.eosToken], false)
    }

    @Test
    func `Bitmask Allows Eos After Completion`() throws {
      let matcher = try self.engine.compile(try genericGrammar())
      let call = genericGrammarText

      for tokenId in encodedGrammarText(call, tokenizer: self.tokenizer) {
        expectNoDifference(matcher.accept(tokenId: tokenId), true)
      }
      expectNoDifference(matcher.isCompleted, true)

      let bitmask = matcher.bitmask()
      expectNoDifference(bitmask[self.eosToken], true)
    }

    @Test
    func `Bitmask Has Expected Size`() throws {
      let matcher = try self.engine.compile(try genericGrammar())
      let bitmask = matcher.bitmask()
      expectNoDifference(bitmask.count, 8192)
    }
  }

  @Suite
  struct `Memory usage tests`: ~Copyable {
    private let engine: XGrammarCompiler
    private let tokenizer: EdgeToolsSPTokenizer

    init() throws {
      let tokenizer = try makeTestTokenizer()
      let engine = try makeGenericXGrammarCompiler(tokenizer: tokenizer)
      self.tokenizer = tokenizer
      self.engine = engine
    }

    @Test
    func `Cache Size Is Zero Before First Compile`() {
      expectNoDifference(self.engine.cacheSizeBytes, 0)
      expectNoDifference(self.engine.cacheLimitBytes, 0)
    }

    @Test
    func `Matcher Reports Non-Zero Memory Size`() throws {
      let matcher = try self.engine.compile(try genericGrammar())
      expectNoDifference(matcher.memorySizeBytes > 0, true)
    }

    @Test
    func `Forked Matcher Reports Equal Memory Size`() throws {
      let matcher = try self.engine.compile(try genericGrammar())
      let forked = try matcher.fork()
      expectNoDifference(forked.memorySizeBytes, matcher.memorySizeBytes)
    }

    @Test
    func `Larger Grammar Has Larger Compiled Matcher`() throws {
      let smallerMatcher = try self.engine.compile(try XGrammarGrammar(literal: "a"))
      let largerMatcher = try self.engine.compile(try genericGrammar())
      expectNoDifference(largerMatcher.memorySizeBytes > smallerMatcher.memorySizeBytes, true)
    }
  }

  private func genericGrammar() throws -> XGrammarGrammar {
    try XGrammarGrammar(literal: genericGrammarText)
  }
#endif
