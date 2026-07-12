#if MLX && canImport(MLX)
  import CustomDump
  import Foundation
  import IssueReporting
  import MLX
  import MLXLMCommon
  import EdgeTools
  import Tokenizers
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXModel tests` {
    private let model: NeedleMLXModel
    private let tokenizer: any Tokenizers.Tokenizer

    init() async throws {
      let url = try await downloadNeedleHF()
      let (tokenizer, model) = try loadNeedleMLXModel(from: url)
      self.tokenizer = tokenizer
      self.model = model
    }

    @Test
    func `TokenIterator Usage`() async throws {
      let grammarEngine = try #require(XGrammarCompiler.needle(tokenizer: self.tokenizer))
      let matcher = try grammarEngine.compile(try XGrammarGrammar.needle(tools: EdgeToolsPrompt.sendAdventureEmail.tools))
      
      var iterator = try TokenIterator(
        input: try LMInput.needle(prompt: .sendAdventureEmail, using: self.tokenizer),
        model: self.model,
        processor: EdgeToolsApplyBitmaskProcessorMLX(matcher: matcher),
        sampler: ArgMaxSampler()
      )

      var tokens = [Int]()
      while let token = iterator.next() {
        tokens.append(token)
        if token == self.tokenizer.eosTokenId || matcher.isTerminated {
          break
        }
      }
      assertSnapshot(of: tokens, as: .dump)
      assertSnapshot(of: self.tokenizer.decode(tokens: tokens), as: .lines)
    }
  }

  extension `NeedleMLXModel tests` {
    fileprivate static let parameters = GenerateParameters(maxTokens: 512, temperature: 0)
  }
#endif
