#if XGrammar
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `XGrammar tests` {
    @Suite
    struct `XGRMatcher tests`: ~Copyable {
      private let engine: XGRCompiler
      private let tokenizer: NeedleSPTokenizer
      private let eosToken: EdgeToolsToken.ID

      init() throws {
        let tokenizer = try testTokenizer()
        let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
        let engine = try makeGenericXGRCompiler(tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.eosToken = eosToken
        self.engine = engine
      }

      @Test
      func `Reset Restores Initial State`() throws {
        let matcher = try self.engine.makeMatcher(try genericGrammar())
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
        let matcher = try self.engine.makeMatcher(try genericGrammar())

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
        let matcher = try self.engine.makeMatcher(try genericGrammar())
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
        let matcher = try self.engine.makeMatcher(try genericGrammar())

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
        let forked = matcher.fork()

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
        let matcher = try self.engine.makeMatcher(try genericGrammar())

        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask[self.eosToken], false)
      }

      @Test
      func `Bitmask Allows Eos After Completion`() throws {
        let matcher = try self.engine.makeMatcher(try genericGrammar())
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
        let matcher = try self.engine.makeMatcher(try genericGrammar())
        let bitmask = matcher.bitmask()
        expectNoDifference(bitmask.count, 8192)
      }
    }

    @Suite
    struct `Constraint tests` {
      @Test
      func `Explicit Grammar Is Returned Unchanged`() throws {
        let toolsGrammar = try XGRGrammar.literal("tool")
        let expectedGrammar = try XGRGrammar.literal("response")
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["tool", "response"],
          vocabularyType: .raw
        )
        let constraint = XGRGenerationConstraint.grammar(expectedGrammar)

        let grammar = try constraint.grammar(
          using: toolsGrammar,
          tokenizerInfo: tokenizerInfo
        )

        expectNoDifference(grammar.ebnf, expectedGrammar.ebnf)
      }

      @Test
      func `Tools Transform Receives The Model Grammar And Tokenizer Info`() throws {
        let toolsGrammar = try XGRGrammar.literal("tool")
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["tool", "<|response|>", ""],
          vocabularyType: .raw,
          stopTokenIDs: [2]
        )
        let constraint = XGRGenerationConstraint.tools { toolsGrammar, tokenizerInfo in
          let responseGrammar = try XGRGrammar.lark(
            "start: <|response|>",
            tokenizerInfo: tokenizerInfo
          )
          return try toolsGrammar.union(responseGrammar)
        }

        let grammar = try constraint.grammar(
          using: toolsGrammar,
          tokenizerInfo: tokenizerInfo
        )
        let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
        let matcher = try compiler.makeMatcher(grammar)

        expectNoDifference(matcher.accept(string: "tool"), true)
      }

      @Test
      func `Unconstrained Uses Universal Grammar`() throws {
        let tokenizer = try testTokenizer()
        let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
        let matcher = try compiler.makeMatcher(
          try XGRGenerationConstraint.unconstrained.grammar(
            using: .universal,
            tokenizerInfo: compiler.tokenizerInfo
          )
        )

        expectNoDifference(matcher.accept(string: "Free form text."), true)
      }
    }

    @Suite(.serialized)
    struct `Memory usage tests`: ~Copyable {
      private let engine: XGRCompiler
      private let tokenizer: NeedleSPTokenizer

      init() throws {
        let tokenizer = try testTokenizer()
        let engine = try makeGenericXGRCompiler(tokenizer: tokenizer)
        self.tokenizer = tokenizer
        self.engine = engine
      }

      @Test
      func `Cache Size Is Zero Before First Compile`() {
        expectNoDifference(self.engine.cacheSizeBytes, 0)
        expectNoDifference(self.engine.cacheLimitBytes, -1)
      }

      @Test
      func `Compiled Grammar Reports Non-Zero Memory Size`() throws {
        let compiledGrammar = try self.engine.compile(try genericGrammar())
        expectNoDifference(compiledGrammar.memorySizeBytes > 0, true)
      }

      @Test
      func `Cache Reports Memory After Compiling`() throws {
        _ = try self.engine.compile(try genericGrammar())
        expectNoDifference(self.engine.cacheSizeBytes > 0, true)
      }

      @Test
      func `Forked Matcher Preserves Compiled Grammar Memory Size`() throws {
        let compiledGrammar = try self.engine.compile(try genericGrammar())
        let matcher = try XGRMatcher(compiledGrammar: compiledGrammar)
        _ = matcher.fork()
        expectNoDifference(compiledGrammar.memorySizeBytes > 0, true)
      }

      @Test
      func `Larger Grammar Has Larger Compiled Grammar`() throws {
        let smallerGrammar = try self.engine.compile(try XGRGrammar.literal("a"))
        let largerGrammar = try self.engine.compile(try genericGrammar())
        expectNoDifference(largerGrammar.memorySizeBytes > smallerGrammar.memorySizeBytes, true)
      }
    }

    @Suite
    struct `XGR serialization tests` {
      @Test
      func `Tokenizer Info Round Trips Through JSON`() throws {
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["a", ""],
          vocabularyType: .raw,
          stopTokenIDs: [1]
        )
        let restoredTokenizerInfo = try XGRTokenizerInfo(
          serializedJSON: try tokenizerInfo.serializedJSON()
        )

        let serializedJSON = try tokenizerInfo.serializedJSON()
        expectNoDifference(try restoredTokenizerInfo.serializedJSON(), serializedJSON)
      }

      @Test
      func `Lark Grammar Matches Its Start Rule`() throws {
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["a", "b", ""],
          vocabularyType: .raw,
          stopTokenIDs: [2]
        )
        let grammar = try XGRGrammar.lark(
          "start: \"a\" | \"b\"",
          tokenizerInfo: tokenizerInfo
        )
        let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
        let matcher = try XGRMatcher(
          compiledGrammar: try compiler.compile(grammar),
          terminateWithoutStopToken: true
        )

        expectNoDifference(matcher.accept(string: "a"), true)
        expectNoDifference(matcher.isTerminated, true)
      }

      @Test
      func `Lark Grammar Resolves Named Special Tokens`() throws {
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["<|tool|>", ""],
          vocabularyType: .raw,
          stopTokenIDs: [1]
        )
        let grammar = try XGRGrammar.lark(
          "start: <|tool|>",
          tokenizerInfo: tokenizerInfo
        )
        let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
        let matcher = try XGRMatcher(
          compiledGrammar: try compiler.compile(grammar),
          terminateWithoutStopToken: true
        )

        expectNoDifference(matcher.accept(tokenId: 0), true)
        expectNoDifference(matcher.isTerminated, true)
      }

      @Test
      func `Lark Grammar Resolves Named Grammar Sources And Objects`() throws {
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["[", "x", "]", ""],
          vocabularyType: .raw,
          stopTokenIDs: [3]
        )
        let item = try XGRGrammar.lark("start: \"x\"", tokenizerInfo: tokenizerInfo)
        let grammar = try XGRGrammar.lark(
          "start: \"[\" @item \"]\" @unused",
          tokenizerInfo: tokenizerInfo,
          namedGrammars: [
            XGRNamedGrammar(name: "item", definition: .grammar(item)),
            XGRNamedGrammar(name: "unused", definition: .lark("start: \"unused\""))
          ]
        )
        let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
        let matcher = try XGRMatcher(
          compiledGrammar: try compiler.compile(grammar),
          terminateWithoutStopToken: true
        )

        expectNoDifference(matcher.accept(string: "[x]unused"), true)
        expectNoDifference(matcher.isTerminated, true)
      }

      @Test
      func `Structural Tag Resolves Token References`() throws {
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["<tool>", "hello", "<end>", ""],
          vocabularyType: .raw,
          stopTokenIDs: [3]
        )
        let grammar = try XGRGrammar.structuralTagJSON(
          #"{"type":"structural_tag","format":{"type":"tag","begin":{"type":"token","token":"<tool>"},"content":{"type":"const_string","value":"hello"},"end":{"type":"token","token":"<end>"}}}"#,
          tokenizerInfo: tokenizerInfo
        )
        let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
        let matcher = try XGRMatcher(
          compiledGrammar: try compiler.compile(grammar),
          terminateWithoutStopToken: true
        )

        expectNoDifference(matcher.accept(tokenId: 0), true)
        expectNoDifference(matcher.accept(tokenId: 1), true)
        expectNoDifference(matcher.accept(tokenId: 2), true)
        expectNoDifference(matcher.isTerminated, true)
      }

      @Test
      func `Grammar And Compiled Grammar Round Trip Through JSON`() throws {
        let tokenizerInfo = try XGRTokenizerInfo(
          encodedVocabulary: ["a", ""],
          vocabularyType: .raw,
          stopTokenIDs: [1]
        )
        let grammar = try XGRGrammar.literal("a")
        let restoredGrammar = try XGRGrammar.serializedJSON(grammar.serializedJSON())
        let ebnf = grammar.ebnf
        expectNoDifference(restoredGrammar.ebnf, ebnf)

        let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
        let compiledGrammar = try compiler.compile(grammar)
        let restoredCompiledGrammar = try XGRCompiledGrammar(
          serializedJSON: try compiledGrammar.serializedJSON(),
          tokenizerInfo: tokenizerInfo
        )
        expectNoDifference(restoredCompiledGrammar.grammar.ebnf, ebnf)
      }

      @Test
      func `Hugging Face Metadata Detects Byte Fallback And Prefix Space`() throws {
        let backendJSON =
          #"{"decoder":{"type":"ByteFallback"},"normalizer":{"type":"Prepend","prepend":"▁"}}"#
        let metadata = try XGRTokenizerInfo.metadata(huggingFaceBackendJSON: backendJSON)
        let tokenizerInfo = try XGRTokenizerInfo.huggingFace(
          encodedVocabulary: ["a", ""],
          backendJSON: backendJSON
        )
        let paddedTokenizerInfo = try XGRTokenizerInfo.huggingFace(
          encodedVocabulary: ["a", ""],
          backendJSON: backendJSON,
          modelVocabularySize: 8
        )

        expectNoDifference(metadata.contains(#""vocab_type":1"#), true)
        expectNoDifference(metadata.contains(#""add_prefix_space":true"#), true)
        expectNoDifference(try tokenizerInfo.serializedJSON().isEmpty, false)
        expectNoDifference(try paddedTokenizerInfo.serializedJSON().contains(#""vocab_size":8"#), true)
      }
    }
  }

  private let genericGrammarText =
    #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

  private func genericGrammar() throws -> XGRGrammar {
    try XGRGrammar.literal(genericGrammarText)
  }
#endif
