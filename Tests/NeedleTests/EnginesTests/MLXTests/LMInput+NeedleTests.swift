#if SwiftNeedleMLX
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
      try self.assertTokensSnapshot(
        system: "You are a helpful assistant who can send emails.",
        user: "Send an email to Henry."
      )
    }

    @Test
    func `Needle Empty System Snapshot`() throws {
      try self.assertTokensSnapshot(
        system: "",
        user: "Send an email to Henry."
      )
    }

    @Test
    func `Needle Empty User Snapshot`() throws {
      try self.assertTokensSnapshot(
        system: "You are a helpful assistant who can send emails.",
        user: ""
      )
    }

    private func assertTokensSnapshot(system: String, user: String) throws {
      let tokenizer = try NeedleSPTokenizer(modelURL: .testTokenizerModel)
      let input = try LMInput.needle(
        prompt: NeedlePrompt(system: system, user: user, tools: [.sendEmail]),
        using: tokenizer
      )
      assertSnapshot(of: input.text.tokens.asArray(Int32.self), as: .dump)
    }
  }
#endif
