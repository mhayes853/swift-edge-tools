#if SwiftNeedleMLX && SwiftNeedleTokenizers && SwiftNeedleSentencepiece
  import CustomDump
  import Foundation
  import MLX
  import MLXLMCommon
  import Needle
  import Testing
  import SnapshotTesting

  @Suite(.serialized, .enabledIfXcode())
  struct `LMInput+Needle tests` {
    @Test
    func `Needle Snapshot`() throws {
      let tokenizer = try NeedleSentencepieceTokenizer(modelURL: .testTokenizerModel)
      let prompt = NeedlePrompt(
        system: "You are a helpful assistant who can send emails.",
        user: "Send an email to Henry.",
        tools: [.sendEmail]
      )
      let input = try LMInput.needle(prompt: prompt, using: tokenizer)
      assertSnapshot(of: input.text.tokens.asArray(Int32.self), as: .dump)
    }
  }
#endif
