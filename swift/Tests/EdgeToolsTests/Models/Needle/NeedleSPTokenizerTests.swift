import CustomDump
import EdgeTools
import Foundation
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

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
  func `Vocabulary Size`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
    expectNoDifference(tokenizer.vocabularySize, 8192)
  }

  @Test
  func `Needle Special Tokens Are Discoverable`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
    let toolsTokenId = tokenizer.needleToolsTokenId
    let toolCallTokenId = tokenizer.needleToolCallTokenId
    guard let toolsTokenId, let toolCallTokenId else {
      Issue.record("Expected Needle special tokens in the vocabulary.")
      return
    }

    expectNoDifference(tokenizer.needleToolsToken, "<tools>")
    expectNoDifference(tokenizer.needleToolCallToken, "<tool_call>")
    expectNoDifference(tokenizer.convertIdToToken(toolsTokenId), "<tools>")
    expectNoDifference(tokenizer.convertIdToToken(toolCallTokenId), "<tool_call>")
  }

  @Test
  func `Throws Error For Invalid URL`() {
    let error = #expect(throws: NeedleSPTokenizerError.self) {
      _ = try NeedleSPTokenizer(modelURL: .temporaryDirectory)
    }
    expectNoDifference(error?.code, .missingModelFile)
  }

  @Test
  func `Encodes Needle Text`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)

    expectNoDifference(tokenizer.encode(text: "This is a test"), [6401, 743, 289, 2959])
  }

  @Test
  func `Encode Then Decode String`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
    let expectedString = "Hello world this is a test"
    let ids = tokenizer.encode(text: expectedString)
    let string = tokenizer.decode(tokens: ids)
    expectNoDifference(string, expectedString)
  }

  #if !os(WASI)
    @Test
    func `Encode Snapshot`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let tokens = tokenizer.encode(text: "This is a test")
      assertSnapshot(of: tokens, as: .dump)
    }
  #endif

  @Test
  func `Encoding Uses Byte Fallback Tokens For Unknown Scalars`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)

    expectNoDifference(tokenizer.encode(text: "🙂"), [8041, 246, 165, 159, 136])
    expectNoDifference(
      tokenizer.convertIdsToTokens([246, 165, 159, 136]),
      ["<0xF0>", "<0x9F>", "<0x99>", "<0x82>"]
    )
  }

  @Test
  func `Decoding Reassembles Byte Fallback Tokens`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)

    expectNoDifference(tokenizer.decode(tokens: [246, 165, 159, 136]), "🙂")
    expectNoDifference(tokenizer.decode(tokens: [2, 246, 165, 159, 136, 1]), "🙂")
    expectNoDifference(tokenizer.decode(tokens: [201]), "�")
  }

  #if !os(WASI)
    @Test
    func `Encodes Long Needle Prompt Snapshot`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let tokens = tokenizer.encode(text: try self.longPrompt())
      let pieces = tokenizer.convertIdsToTokens(tokens)
      let annotatedTokens = zip(tokens, pieces)
        .map { token, piece in "\(token)\t\(String(reflecting: piece ?? ""))" }
        .joined(separator: "\n")

      assertSnapshot(of: annotatedTokens, as: .lines)
    }

    @Test
    func `Decodes Long Needle Prompt Snapshot`() throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
      let prompt = try self.longPrompt()
      let decoded = tokenizer.decode(tokens: tokenizer.encode(text: prompt))

      assertSnapshot(of: decoded, as: .lines)
      expectNoDifference(decoded, prompt)
    }
  #endif

  @Test(
    arguments: [
      ("", [Int](), ""),
      (" Hello  world ", [7318, 363, 5338, 745], "Hello world"),
      ("<tools>x<tool_call>", [8041, 5, 8073, 4], "<tools>x<tool_call>"),
      ("🙂", [8041, 246, 165, 159, 136], "🙂"),
      ("é", [5116], "é"),
      ("é", [341, 210, 135], "é"),
      ("\tfoo\nbar", [8041, 15, 8060, 377, 16, 8064, 286], "\tfoo\nbar"),
      ("a<tools>b", [289, 5, 8064], "a<tools>b")
    ]
  )
  func `Encodes And Decodes SentencePiece Edge Cases`(
    text: String,
    expectedTokens: [Int],
    expectedDecodedText: String
  ) throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
    let tokens = tokenizer.encode(text: text)

    expectNoDifference(tokens, expectedTokens)
    expectNoDifference(tokenizer.decode(tokens: tokens), expectedDecodedText)
  }

  @Test
  func `Token Id Conversions`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
    let expectedTokens = ["string", "▁the"]

    let tokenIds = expectedTokens.compactMap { tokenizer.convertTokenToId($0) }
    expectNoDifference(tokenIds, [315, 302])

    let tokens = tokenIds.compactMap { tokenizer.convertIdToToken($0) }
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
  func `HF Convert Id To Token Returns Unknown For Unknown Token Id`() throws {
    let tokenizer = try NeedleSPTokenizer(modelURL: self.modelURL)
    expectNoDifference(tokenizer.convertIdToToken(3), tokenizer.unknownToken)
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
    expectNoDifference(error?.code, .invalidProtobuf)
  }

  @Test
  func `Init From Empty Data Throws`() {
    #expect(throws: NeedleSPTokenizerError.self) {
      _ = try NeedleSPTokenizer(data: Data())
    }
  }

  @Test
  func `Rejects Non BPE Model`() {
    let model = self.minimalModel(modelType: 1, byteFallback: false)

    #expect(throws: NeedleSPTokenizerError.self) {
      _ = try NeedleSPTokenizer(data: model)
    }
  }

  @Test
  func `Rejects Incomplete Byte Fallback Vocabulary`() {
    let model = self.minimalModel(modelType: 2, byteFallback: true)

    #expect(throws: NeedleSPTokenizerError.self) {
      _ = try NeedleSPTokenizer(data: model)
    }
  }

  @Test
  func `Allows Unused Pieces`() throws {
    let unusedPiece = [UInt8]([0x0A, 0x08] + Array("<unused>".utf8) + [0x18, 0x05])
    let model = self.minimalModel(modelType: 2, byteFallback: false)
      + [0x0A, UInt8(unusedPiece.count)] + unusedPiece
    let tokenizer = try NeedleSPTokenizer(data: model)

    expectNoDifference(tokenizer.convertIdToToken(1), "<unused>")
    expectNoDifference(tokenizer.decode(tokens: [1]), "")
  }

  private func longPrompt() throws -> String {
    try NeedlePrompt(
      system: """
        You are a careful operations assistant. Choose the smallest applicable tool, preserve exact
        identifiers, and explain any ambiguity before taking an irreversible action. Unicode input
        such as café, é, 東京, and 🙂 must remain intact across tokenization.
        """,
      user: """
        Draft an email to ops@example.com summarizing ticket ABC-42, check the weather in Reykjavík,
        and prepare a dry-run configuration with labels, routing, tuple arguments, and nested flags.
        Do not send anything until the configuration has been validated.
        """
    )
    .formatted(tools: [.sendEmail, .getWeather, .complexTool])
  }

  private func minimalModel(modelType: UInt8, byteFallback: Bool) -> [UInt8] {
    let unknownPiece = [UInt8](
      [0x0A, 0x05] + Array("<unk>".utf8) + [0x18, 0x02]
    )
    var trainer = [0x18, modelType]
    if byteFallback {
      trainer += [0x98, 0x02, 0x01]
    }
    return [0x0A, UInt8(unknownPiece.count)] + unknownPiece
      + [0x12, UInt8(trainer.count)] + trainer
  }
}
