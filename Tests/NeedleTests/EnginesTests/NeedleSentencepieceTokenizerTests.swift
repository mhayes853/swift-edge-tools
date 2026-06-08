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
      let tokenizer = try NeedleSentencepieceTokenizer(modelURL: self.modelURL)
      expectNoDifference(tokenizer.unkTokenId, 3)
      expectNoDifference(tokenizer.bosTokenId, 2)
      expectNoDifference(tokenizer.eosTokenId, 1)
      expectNoDifference(tokenizer.padTokenId, 0)
    }

    @Test
    func `Throws Error For Invalid URL`() {
      #expect(throws: NeedleSentencepieceTokenizerError.failedToLoad) {
        _ = try NeedleSentencepieceTokenizer(modelURL: .temporaryDirectory)
      }
    }

    @Test
    func `Encode Then Decode String`() throws {
      let tokenizer = try NeedleSentencepieceTokenizer(modelURL: self.modelURL)
      let expectedString = "Hello world this is a test"
      let ids = tokenizer.encode(text: expectedString)
      let string = tokenizer.decode(tokenIds: ids)
      expectNoDifference(string, expectedString)
    }

    @Test
    func `Encode Snapshot`() throws {
      let tokenizer = try NeedleSentencepieceTokenizer(modelURL: self.modelURL)
      let tokens = tokenizer.encode(text: "This is a test")
      assertSnapshot(of: tokens, as: .dump)
    }
  }
#endif
