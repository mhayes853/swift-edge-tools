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
      let url = try await downloadNeedle()
      var tokenizer = try self.tokenizer(url: url)
      let model = try loadEdgeToolsLanguageModel(
        NeedleMLXModel.self,
        from: url,
        model: { NeedleMLXModel(configuration: $0) }
      )
      let tokenizerInfo = try XGRTokenizerInfo.needle(tokenizer: tokenizer)
      let grammarEngine = try XGRCompiler(tokenizerInfo: tokenizerInfo)
      let grammar = try XGRGrammar.needle(tools: [.sendEmail])
      let compiledGrammar = try grammarEngine.compile(grammar)
      let matcher = try XGRMatcher(compiledGrammar: compiledGrammar)

      tokenizer = try self.tokenizer(url: url)

      var iterator = try TokenIterator(
        input: try LMInput.needle(
          prompt: .sendAdventureEmail,
          tools: [.sendEmail],
          using: tokenizer
        ),
        model: model,
        processor: EdgeToolsApplyBitmaskProcessorMLX(matcher: matcher),
        sampler: ArgMaxSampler()
      )

      tokenizer = try self.tokenizer(url: url)
      var tokens = [Int]()
      while let token = iterator.next() {
        tokens.append(token)
        if token == tokenizer.eosTokenId {
          break
        }
      }
      assertSnapshot(of: tokens, as: .dump)
      assertSnapshot(of: tokenizer.decode(tokens: tokens), as: .lines)
    }

    private func tokenizer(url: URL) throws -> NeedleSPTokenizer {
      try NeedleSPTokenizer(modelURL: url.appending(path: "tokenizer.model"))
    }
  }

  extension `NeedleMLXModel tests` {
    fileprivate static let parameters = GenerateParameters(maxTokens: 512, temperature: 0)
  }
#endif
