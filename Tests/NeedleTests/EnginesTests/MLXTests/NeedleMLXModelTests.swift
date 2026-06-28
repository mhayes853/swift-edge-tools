#if SwiftNeedleMLX
  import CustomDump
  import Foundation
  import IssueReporting
  import MLX
  import MLXLMCommon
  import Needle
  import Tokenizers
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXModel tests` {
    private let model: NeedleMLXModel
    private let tokenizer: any Tokenizers.Tokenizer

    init() async throws {
      let url = try await downloadNeedleHF()
      let engine = try NeedleMLXEngine(from: url)
      self.model = engine.model
      self.tokenizer = engine.tokenizer
    }

    @Test
    func `TokenIterator Usage`() async throws {
      var iterator = try TokenIterator(
        input: try LMInput.needle(prompt: Self.basePrompt, using: self.tokenizer),
        model: self.model,
        parameters: Self.parameters
      )

      var tokens = [Int]()
      while let token = iterator.next() {
        tokens.append(token)
        if token == self.tokenizer.eosTokenId {
          break
        }
      }
      assertSnapshot(of: tokens, as: .dump)
      assertSnapshot(of: self.tokenizer.decode(tokens: tokens), as: .lines)
    }
  }

  extension `NeedleMLXModel tests` {
    fileprivate static let basePrompt = NeedlePrompt(
      system: "",
      user: "Send an email to Henry asking him to go on an adventure.",
      tools: [.sendEmail]
    )

    fileprivate static let parameters = GenerateParameters(maxTokens: 512, temperature: 0)
  }
#endif
