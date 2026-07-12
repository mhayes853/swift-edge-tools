#if MLX && canImport(MLX)
  import CustomDump
  import Foundation
  import IssueReporting
  import MLX
  import MLXLMCommon
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXModel tests` {
    @Test
    func `TokenIterator Usage`() async throws {
      let url = try await downloadNeedleHF()
      var tokenizer = try self.tokenizer(url: url)
      let model = try loadNeedleMLXModel(from: url)
      let grammarEngine = XGrammarCompiler.needle(tokenizer: consume tokenizer)!
      let grammar = try XGrammarGrammar.needle(tools: EdgeToolsPrompt.sendAdventureEmail.tools)
      let matcher = try grammarEngine.compile(grammar)
      
      tokenizer = try self.tokenizer(url: url)

      var iterator = try TokenIterator(
        input: try LMInput.needle(prompt: .sendAdventureEmail, using: consume tokenizer),
        model: model,
        processor: EdgeToolsApplyBitmaskProcessorMLX(matcher: matcher),
        sampler: ArgMaxSampler()
      )
      
      tokenizer = try self.tokenizer(url: url)
      var tokens = [Int]()
      while let token = iterator.next() {
        tokens.append(token)
        if token == tokenizer.eosTokenId || matcher.isTerminated {
          break
        }
      }
      assertSnapshot(of: tokens, as: .dump)
      assertSnapshot(of: tokenizer.decode(tokens: tokens), as: .lines)
    }
    
    private func tokenizer(url: URL) throws -> EdgeToolsSPTokenizer {
      try EdgeToolsSPTokenizer(modelURL: url.appending(path: "tokenizer.model"))
    }
  }

  extension `NeedleMLXModel tests` {
    fileprivate static let parameters = GenerateParameters(maxTokens: 512, temperature: 0)
  }
#endif
