#if Sentencepiece
  import Needle
  import Tokenizers
  import Testing
  import CustomDump
  import Foundation
  import SnapshotTesting

  @Suite
  struct `NeedleSPTokenizer tests` {
    private let modelURL = Bundle.module.url(forResource: "test_tokenizer", withExtension: "model")!

    @Test
    func `Sentinel Token Ids`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.unknownTokenId, 3)
      expectNoDifference(tokenizer.bosTokenId, 2)
      expectNoDifference(tokenizer.eosTokenId, 1)
    }

    @Test
    func `Needle Special Tokens Are Discoverable`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let toolsTokenId = try #require(tokenizer.toolsTokenId)
      let toolCallTokenId = try #require(tokenizer.toolCallTokenId)

      expectNoDifference(tokenizer.toolsToken, "<tools>")
      expectNoDifference(tokenizer.toolCallToken, "<tool_call>")
      expectNoDifference(tokenizer.convertIdToToken(toolsTokenId), "<tools>")
      expectNoDifference(tokenizer.convertIdToToken(toolCallTokenId), "<tool_call>")
    }

    @Test
    func `Throws Error For Invalid URL`() {
      let error = #expect(throws: NeedleSPTokenizerError.self) {
        _ = try NeedleSPTokenizer(modelURL: .temporaryDirectory)
      }
      expectNoDifference(error?.message.lowercased().contains("file not found"), true)
    }

    @Test
    func `Encode Then Decode String`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let expectedString = "Hello world this is a test"
      let ids = tokenizer.encode(text: expectedString)
      let string = tokenizer.decode(tokens: ids)
      expectNoDifference(string, expectedString)
    }

    @Test
    func `Decode Skips Needle Special Tokens`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let toolsTokenId = try #require(tokenizer.toolsTokenId)
      let toolCallTokenId = try #require(tokenizer.toolCallTokenId)

      expectNoDifference(
        tokenizer.decode(tokens: [toolsTokenId, toolCallTokenId], skipSpecialTokens: true),
        ""
      )
    }

    @Test
    func `Encode Snapshot`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let tokens = tokenizer.encode(text: "This is a test")
      assertSnapshot(of: tokens, as: .dump)
    }

    @Test
    func `Token Id Conversions`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)

      let expectedTokens = ["string", "▁the"]

      let tokenIds = expectedTokens.compactMap(tokenizer.convertTokenToId)
      expectNoDifference(tokenIds, [315, 302])

      let tokens = tokenIds.compactMap(tokenizer.convertIdToToken)
      expectNoDifference(tokens, expectedTokens)
    }

    @Test
    func `Unk Token Id For Non-Existent Token`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.convertTokenToId("shdkjhdksahdiiwsubdnuiwsduybsw"), nil as Int?)
    }

    @Test
    func `Empty Token For Non-Existent Token Id`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.convertIdToToken(287_399_329), nil as String?)
    }

    @Test
    func `Apply Chat Template Throws Missing Chat Template`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let error = #expect(throws: TokenizerError.self) {
        _ = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Hello"]])
      }
      expectNoDifference(
        error?.errorDescription,
        TokenizerError.missingChatTemplate.errorDescription
      )
    }

    @Test
    func `HF Unknown Token String Is Not Nil`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.unknownToken, "<unk>")
    }

    @Test
    func `HF Convert Token To Id Returns Unk For Unk Token`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.convertTokenToId("<unk>"), tokenizer.unknownTokenId)
    }

    @Test
    func `HF Convert Token To Id Returns Nil For Non Existent Token`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(
        tokenizer.convertTokenToId("shdkjhdksahdiiwsubdnuiwsduybsw"),
        nil as Int?
      )
    }

    @Test
    func `HF Convert Id To Token Returns Unkk For Unknown Token Id`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.convertIdToToken(3), tokenizer.unknownToken)
    }

    @Test
    func `HF Convert Id To Token Returns Nil For Non Existent Token Id`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.convertIdToToken(287_399_329), nil as String?)
    }

    @Test
    func `Init From Data Produces Equivalent Tokenizer To Init From URL`() throws {
      let data = try Data(contentsOf: self.modelURL)
      let urlTokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let dataTokenizer = try NeedleSPTokenizer(data: data)

      expectNoDifference(dataTokenizer.unknownTokenId, urlTokenizer.unknownTokenId)
      expectNoDifference(dataTokenizer.bosTokenId, urlTokenizer.bosTokenId)
      expectNoDifference(dataTokenizer.eosTokenId, urlTokenizer.eosTokenId)
      expectNoDifference(dataTokenizer.eosToken, urlTokenizer.eosToken)
      expectNoDifference(dataTokenizer.bosToken, urlTokenizer.bosToken)
      expectNoDifference(dataTokenizer.unknownToken, urlTokenizer.unknownToken)

      let text = "Hello world this is a test"
      expectNoDifference(
        dataTokenizer.encode(text: text),
        urlTokenizer.encode(text: text)
      )
    }

    @Test
    func `Init From Data Throws On Garbage Bytes`() {
      let error = #expect(throws: NeedleSPTokenizerError.self) {
        _ = try NeedleSPTokenizer(data: Data([0x00, 0x01, 0x02, 0x03]))
      }
      expectNoDifference(error?.message.isEmpty, false)
    }

    @Test
    func `Init From Empty Data Throws`() {
      #expect(throws: NeedleSPTokenizerError.self) {
        _ = try NeedleSPTokenizer(data: Data())
      }
    }
  }
#endif
