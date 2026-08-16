#if FoundationEssentials
  import CustomDump
  import EdgeToolsTokenizers
  import Foundation
  import Testing

  @Suite
  struct `EdgeToolsAutoTokenizer tests` {
    @Test
    func `Missing Tokenizer Describes Available Options`() async {
      let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
      defer { try? FileManager.default.removeItem(at: directory) }

      let error = await #expect(throws: EdgeToolsError.self) {
        _ = try await EdgeToolsAutoTokenizer.from(modelDirectory: directory)
      }
      expectNoDifference(error?.code, .noCompatibleTokenizer)
      expectNoDifference(error?.message.contains("tokenizer.json"), true)
    }

    #if !HuggingFaceTokenizers
      @Test
      func `Reports The Missing Trait When A Tokenizer Cannot Be Read`() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("{}".utf8).write(to: directory.appending(path: "tokenizer.json"))

        let error = await #expect(throws: EdgeToolsError.self) {
          _ = try await EdgeToolsAutoTokenizer.from(modelDirectory: directory)
        }
        expectNoDifference(error?.code, .noCompatibleTokenizer)
        expectNoDifference(error?.message.contains("HuggingFaceTokenizers"), true)
      }
    #endif

    #if HuggingFaceTokenizers
      @Test
      func `Loads Hugging Face Tokenizer JSON`() async throws {
        let directory = try self.makeHuggingFaceTokenizerDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try await EdgeToolsAutoTokenizer.from(modelDirectory: directory)
        let preTrainedTokenizer = try #require(tokenizer as? HuggingFaceTokenizer)
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

      private func makeHuggingFaceTokenizerDirectory() throws -> URL {
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
