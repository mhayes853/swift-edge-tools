#if SwiftNeedleSentencepiece
  import Needle
  import Testing
  import CustomDump
  import Foundation
  import SnapshotTesting

  @Suite
  struct `NeedleSentencepieceTokenizer tests` {
    private let modelURL = Bundle.module.url(forResource: "test_tokenizer", withExtension: "model")!

    @Test
    func `Sentinel Token Ids`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
      expectNoDifference(tokenizer.unknownTokenId, 3)
      expectNoDifference(tokenizer.bosTokenId, 2)
      expectNoDifference(tokenizer.eosTokenId, 1)
      expectNoDifference(tokenizer.padTokenId, 0)
    }

    @Test
    func `Throws Error For Invalid URL`() {
      let error = #expect(throws: NeedleSentencepieceTokenizerError.self) {
        _ = try NeedleSPTokenizingModel(modelURL: .temporaryDirectory)
      }
      expectNoDifference(error?.message.lowercased().contains("file not found"), true)
    }

    @Test
    func `Encode Then Decode String`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
      let expectedString = "Hello world this is a test"
      let ids = tokenizer.encode(text: expectedString)
      let string = tokenizer.decode(tokenIds: ids)
      expectNoDifference(string, expectedString)
    }

    @Test
    func `Encode Snapshot`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
      let tokens = tokenizer.encode(text: "This is a test")
      assertSnapshot(of: tokens, as: .dump)
    }

    @Test
    func `Encoded Vocab Snapshot`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
      let vocab = tokenizer.encodedVocab()
      assertSnapshot(of: vocab, as: .dump)
    }

    @Test
    func `Token Id Conversions`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)

      let expectedTokens = ["string", "▁the"]

      let tokenIds = tokenizer.tokenIds(from: expectedTokens)
      expectNoDifference(tokenIds, [315, 302])

      let tokens = tokenizer.tokens(from: tokenIds)
      expectNoDifference(tokens, expectedTokens)
    }

    @Test
    func `Unk Token Id For Non-Existent Token`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
      expectNoDifference(
        tokenizer.tokenIds(from: ["shdkjhdksahdiiwsubdnuiwsduybsw"]),
        [tokenizer.unknownTokenId]
      )
    }

    @Test
    func `Empty Token For Non-Existent Token Id`() throws {
      let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
      expectNoDifference(tokenizer.tokens(from: [287_399_329]), [""])
    }

    #if SwiftNeedleTokenizers
      @Test
      func `HF Unknown Token String Is Not Nil`() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
        expectNoDifference(tokenizer.unknownToken, "<unk>")
      }

      @Test
      func `HF Convert Token To Id Returns Unk For Unk Token`() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
        expectNoDifference(tokenizer.convertTokenToId("<unk>"), tokenizer.unknownTokenId)
      }

      @Test
      func `HF Convert Token To Id Returns Nil For Non Existent Token`() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
        expectNoDifference(
          tokenizer.convertTokenToId("shdkjhdksahdiiwsubdnuiwsduybsw"),
          nil as Int?
        )
      }

      @Test
      func `HF Convert Id To Token Returns Nil For Unknown Token Id`() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
        expectNoDifference(tokenizer.convertIdToToken(3), nil as String?)
      }

      @Test
      func `HF Convert Id To Token Returns Nil For Non Existent Token Id`() throws {
        let tokenizer = try NeedleSPTokenizingModel(modelURL: self.modelURL)
        expectNoDifference(tokenizer.convertIdToToken(287_399_329), nil as String?)
      }
    #endif
  }
#endif
