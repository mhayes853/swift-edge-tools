#if Foundation
  import _EdgeToolsFoundation
#endif

#if Transformers && canImport(Tokenizers)
  import Tokenizers
#endif

// MARK: - EdgeToolsAutoTokenizer

public enum EdgeToolsAutoTokenizer {
  #if Foundation
    public static func from(
      modelDirectory directoryURL: URL
    ) async throws -> sending any EdgeToolsTokenizer {
      let transformersURL = directoryURL.appending(path: "tokenizer.json")
      let hasTransformersTokenizer = FileManager.default.fileExists(
        atPath: transformersURL.path()
      )
      var failures = [String]()

      #if Transformers && canImport(Tokenizers)
        if hasTransformersTokenizer {
          do {
            return try await loadTransformersTokenizer(
              from: directoryURL,
              tokenizerURL: transformersURL
            )
          } catch {
            failures.append("tokenizer.json could not be loaded: \(error)")
          }
        }
      #endif

      throw EdgeToolsError.noCompatibleTokenizer(
        in: directoryURL.path(),
        hasTransformersTokenizer: hasTransformersTokenizer,
        failures: failures
      )
    }
  #endif
}

#if Transformers && Foundation && canImport(Tokenizers)
  private func loadTransformersTokenizer(
    from directoryURL: URL,
    tokenizerURL: URL
  ) async throws -> sending any EdgeToolsTokenizer {
    let tokenizer = try await AutoTokenizer.from(modelFolder: directoryURL)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
      throw EdgeToolsError(
        code: .unsupportedTransformersTokenizer,
        message: "swift-transformers created an unsupported tokenizer type from \(tokenizerURL)."
      )
    }
    let backendJSON = try loadHuggingFaceBackendJSON(from: tokenizerURL)

    return TransformersTokenizer(
      tokenizer: tokenizer,
      backendJSON: backendJSON
    )
  }
#endif
