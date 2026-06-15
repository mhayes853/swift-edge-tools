#if SwiftNeedleXGrammar && SwiftNeedleSentencepiece
  import CustomDump
  import Foundation
  import Needle
  import Testing

  @Suite
  struct `NeedleXGrammarEngine tests` {
    private let engine: NeedleXGrammarEngine
    private let tokenizer: NeedleSentencepieceTokenizer

    init() throws {
      let tokenizer = try NeedleSentencepieceTokenizer(modelURL: .testTokenizerModel)
      self.tokenizer = tokenizer
      self.engine = NeedleXGrammarEngine(
        encodedVocab: tokenizer.encodedVocab(),
        eosTokenId: try #require(tokenizer.eosTokenId)
      )
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
    struct `Matcher tests` {
      private let engine: NeedleXGrammarEngine
      private let tokenizer: NeedleSentencepieceTokenizer
      private let eosToken: NeedleToken.ID

      init() throws {
        let tokenizer = try NeedleSentencepieceTokenizer(modelURL: .testTokenizerModel)
        self.tokenizer = tokenizer
        self.eosToken = try #require(tokenizer.eosTokenId)
        self.engine = NeedleXGrammarEngine(
          encodedVocab: tokenizer.encodedVocab(),
          eosTokenId: self.eosToken
        )
      }

      @Test
      func `Reset Restores Initial State`() async throws {
        let matcher = try await self.engine.compile(tools: [.sendEmail])

        for tokenId in self.tokenizer.encode(text: "hello world") {
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
        let matcher = try await self.engine.compile(tools: [.sendEmail])

        expectNoDifference(matcher.isCompleted, false)
        expectNoDifference(matcher.isTerminated, false)

        for tokenId in self.tokenizer.encode(text: "hello world") {
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
        let matcher = try await self.engine.compile(tools: [.sendEmail])

        let tokens = self.tokenizer.encode(text: "hello world")
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
        let matcher = try await self.engine.compile(tools: [.sendEmail])

        for tokenId in self.tokenizer.encode(text: "hello world") {
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

        let call = """
          <tool_call>[{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1,"BETA_LABEL":2},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
          """

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
      }

      @Test
      func `Accepts Valid Simple Tool Call`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

        let call = #"<tool_call>[{"name":"get_weather","arguments":{"location":"Seoul"}}]"#

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
      }

      @Test
      func `Accepts Multiple Consecutive Tool Calls`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

        let calls =
          #"<tool_call>[{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":"get_weather","arguments":{"location":"Henry's Altar"}}]"#

        for tokenId in self.tokenizer.encode(text: calls) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), true)
      }

      @Test
      func `Rejects Missing Required Field`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call = """
          <tool_call>[{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"]}}]
          """

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), false)
      }

      @Test
      func `Rejects Wrong Type`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call = """
          <tool_call>[{"name":"complex_tool","arguments":{"title":"alpha","count":"3.5","enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
          """

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), false)
      }

      @Test
      func `Rejects Invalid Pattern Properties`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call = """
          <tool_call>[{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"alpha":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
          """

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), false)
      }

      @Test
      func `Rejects Extra Property`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call = """
          <tool_call>[{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]},"extra":1}}]
          """

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), false)
      }

      @Test
      func `Rejects Malformed JSON`() async throws {
        let matcher = try await self.engine.compile(tools: [.complexTool])

        let call = """
          <tool_call>[{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],"config":{"threshold":0.75,"flags":[true,false]}}]
          """

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), false)
      }

      @Test
      func `Rejects Invalid Tool Name`() async throws {
        let matcher = try await self.engine.compile(tools: [.getWeather])

        let call = #"<tool_call>[{"name":"not_a_real_tool","arguments":{"location":"Seoul"}}]"#

        for tokenId in self.tokenizer.encode(text: call) {
          expectNoDifference(matcher.accept(tokenId: tokenId), true)
        }
        expectNoDifference(matcher.accept(tokenId: self.eosToken), false)
      }
    }
  }

  private enum NeedleXGrammarEngineTestError: Error {
    case missingEosToken
  }
#endif
