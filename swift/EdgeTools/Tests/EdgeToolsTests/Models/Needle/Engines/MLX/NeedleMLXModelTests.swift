#if MLX && canImport(MLX) && !os(WASI)
  import CustomDump
  import EdgeTools
  import Foundation
  import MLX
  import MLXLMCommon
  import Testing

  @Suite(.serialized, .enabledIfMLXTests())
  struct `NeedleMLXProfile tests` {
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
#endif
