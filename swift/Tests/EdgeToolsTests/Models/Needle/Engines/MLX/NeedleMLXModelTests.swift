#if MLX && canImport(MLX) && !os(WASI)
  import CustomDump
  import Foundation
  import IssueReporting
  import MLX
  import MLXLMCommon
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXProfile tests` {
    @Test
    @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `TokenIterator Usage`() async throws {
      let url = try await downloadNeedle()
      let directory = MLXModelDirectory(url: url)
      var tokenizer = try self.tokenizer(url: url)
      let configuration = try directory.loadConfiguration(NeedleModelConfiguration.self)
      let profile = NeedleMLXProfile(configuration: configuration)
      try profile.loadWeights(from: directory)
      let model = profile.languageModel
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
        processor: MLXBitmaskProcessor(matcher: matcher),
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

    @Test
    func `KV Cache Grows Past Initial Capacity`() async throws {
      let url = try await downloadNeedle()
      let directory = MLXModelDirectory(url: url)
      let tokenizer = try self.tokenizer(url: url)
      let configuration = try directory.loadConfiguration(NeedleModelConfiguration.self)
      let profile = NeedleMLXProfile(configuration: configuration)
      try profile.loadWeights(from: directory)
      let model = profile.languageModel
      let input = try LMInput.needle(
        prompt: .sendAdventureEmail,
        tools: [.sendEmail],
        using: tokenizer
      )
      let cache = model.newCache(parameters: nil)
      let prepared = try model.prepare(input.text.tokens, cache: cache, windowSize: nil)
      var output = try #require(prepared)
      let token = LMInput.Text(tokens: MLXArray([4]))[text: .newAxis]

      for _ in 0..<256 {
        output = model(token, cache: cache, state: output.state)
      }
      eval(cache)
      expectNoDifference(cache.map(\.offset), [Int](repeating: 257, count: cache.count))
    }

    private func tokenizer(url: URL) throws -> NeedleSPTokenizer {
      try NeedleSPTokenizer(modelURL: url.appending(path: "tokenizer.model"))
    }
  }

  extension `NeedleMLXProfile tests` {
    fileprivate static let parameters = GenerateParameters(maxTokens: 512, temperature: 0)
  }
#endif
