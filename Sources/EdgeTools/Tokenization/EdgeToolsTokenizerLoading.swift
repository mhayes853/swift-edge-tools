import Foundation

#if Transformers
  import Tokenizers
#endif

package func loadEdgeToolsTokenizer(
  from directoryURL: URL
) async throws -> sending any EdgeToolsTokenizer & ~Copyable {
  let fileManager = FileManager.default
  let sentencepieceURL = directoryURL.appending(path: "tokenizer.model")
  let transformersURL = directoryURL.appending(path: "tokenizer.json")

  #if Sentencepiece
    if fileManager.fileExists(atPath: sentencepieceURL.path()) {
      return try EdgeToolsSPTokenizer(modelURL: sentencepieceURL)
    }
  #endif

  #if Transformers
    if fileManager.fileExists(atPath: transformersURL.path()) {
      return try await loadTransformersTokenizer(
        from: directoryURL,
        tokenizerURL: transformersURL
      )
    }
  #endif

  throw EdgeToolsTokenizerLoadingError.noCompatibleTokenizer(
    in: directoryURL,
    hasSentencepieceModel: fileManager.fileExists(atPath: sentencepieceURL.path()),
    hasTransformersTokenizer: fileManager.fileExists(atPath: transformersURL.path())
  )
}

public struct EdgeToolsTokenizerLoadingError: Hashable, Sendable, Error {
  public let message: String

  public static func noCompatibleTokenizer(
    in directoryURL: URL,
    hasSentencepieceModel: Bool,
    hasTransformersTokenizer: Bool
  ) -> Self {
    var details = [String]()

    #if Sentencepiece
      details.append("Sentencepiece is enabled, but tokenizer.model was not found.")
    #else
      if hasSentencepieceModel {
        details.append(
          "tokenizer.model exists, but the Sentencepiece trait is not enabled. Enable Sentencepiece to load it."
        )
      } else {
        details.append(
          "The Sentencepiece trait is not enabled; enable it to search for tokenizer.model."
        )
      }
    #endif

    #if Transformers
      details.append("Transformers is enabled, but tokenizer.json was not found.")
    #else
      if hasTransformersTokenizer {
        details.append(
          "tokenizer.json exists, but the Transformers trait is not enabled. Enable Transformers to load it."
        )
      } else {
        details.append(
          "The Transformers trait is not enabled; enable it to search for tokenizer.json."
        )
      }
    #endif

    #if !Sentencepiece && !Transformers
      details.append(
        "Enable at least one tokenizer trait: Sentencepiece for tokenizer.model or Transformers for tokenizer.json."
      )
    #endif

    return Self(
      message:
        "No compatible tokenizer was found in \(directoryURL.path()). \(details.joined(separator: " "))"
    )
  }

  public static func unsupportedTransformersTokenizer(at tokenizerURL: URL) -> Self {
    Self(
      message:
        "swift-transformers created an unsupported tokenizer type from \(tokenizerURL.path())."
    )
  }
}

#if Transformers
  private func loadTransformersTokenizer(
    from directoryURL: URL,
    tokenizerURL: URL
  ) async throws -> sending any EdgeToolsTokenizer & ~Copyable {
    let tokenizer = try await AutoTokenizer.from(modelFolder: directoryURL)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
      throw EdgeToolsTokenizerLoadingError.unsupportedTransformersTokenizer(at: tokenizerURL)
    }
    return tokenizer
  }
#endif
