#if Foundation
  import CustomDump
  import EdgeTools
  import Foundation
  import Testing

  @Suite
  struct `EdgeToolsTokenizerLoading tests` {
    @Test
    func `Missing Tokenizer Describes Available Options`() async {
      let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: directory) }

      let error = await #expect(throws: EdgeToolsError.self) {
        _ = try await loadEdgeToolsTokenizer(from: directory)
      }
      expectNoDifference(error?.code, .noCompatibleTokenizer)
      expectNoDifference(error?.message.contains("tokenizer.model"), true)
      expectNoDifference(error?.message.contains("tokenizer.json"), true)
    }

    @Test
    func `Loads SentencePiece Tokenizer Model`() async throws {
      let fileManager = FileManager.default
      let directory = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
      try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
      defer { try? fileManager.removeItem(at: directory) }
      try fileManager.copyItem(
        at: .testTokenizerModel,
        to: directory.appending(path: "tokenizer.model")
      )

      await #expect(throws: Never.self) {
        _ = try await loadEdgeToolsTokenizer(from: directory)
      }
    }

    #if Transformers
      @Test
      func `Loads Transformers Tokenizer JSON`() async throws {
        let directory = try self.makeTransformersTokenizerDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try await loadEdgeToolsTokenizer(from: directory)
        let preTrainedTokenizer = try #require(tokenizer as? TransformersTokenizer)
        let tokenIDs = (
          preTrainedTokenizer.bosTokenId,
          preTrainedTokenizer.eosTokenId,
          preTrainedTokenizer.unknownTokenId
        )
        expectNoDifference(tokenIDs.0, 0)
        expectNoDifference(tokenIDs.1, 2)
        expectNoDifference(tokenIDs.2, 3)
        expectNoDifference(preTrainedTokenizer.backendJSON.contains("\"model\""), false)
      }

      private func makeTransformersTokenizerDirectory() throws -> URL {
        let fileManager = FileManager.default
        let directory = fileManager.temporaryDirectory.appending(path: UUID().uuidString)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

        let tokenizerURL = Bundle.module.url(
          forResource: "test_tokenizer",
          withExtension: "json"
        )!
        try fileManager.copyItem(
          at: tokenizerURL,
          to: directory.appending(path: "tokenizer.json")
        )

        let configurationURL = Bundle.module.url(
          forResource: "test_tokenizer_config",
          withExtension: "json"
        )!
        try fileManager.copyItem(
          at: configurationURL,
          to: directory.appending(path: "tokenizer_config.json")
        )
        return directory
      }
    #endif
  }
#endif
