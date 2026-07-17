import Foundation

#if Transformers
  import Tokenizers
#endif

package func loadEdgeToolsTokenizer(
  from directoryURL: URL
) async throws -> sending any EdgeToolsTokenizer {
  let fileManager = FileManager.default
  let sentencePieceURL = directoryURL.appending(path: "tokenizer.model")
  let transformersURL = directoryURL.appending(path: "tokenizer.json")
  let hasSentencePieceModel = fileManager.fileExists(atPath: sentencePieceURL.path())
  let hasTransformersTokenizer = fileManager.fileExists(atPath: transformersURL.path())
  var failures = [String]()

  if hasSentencePieceModel {
    do {
      return try NeedleSPTokenizer(modelURL: sentencePieceURL)
    } catch {
      failures.append("tokenizer.model could not be loaded: \(error)")
    }
  }

  #if Transformers
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

  throw EdgeToolsTokenizerLoadingError.noCompatibleTokenizer(
    in: directoryURL,
    hasSentencePieceModel: hasSentencePieceModel,
    hasTransformersTokenizer: hasTransformersTokenizer,
    failures: failures
  )
}

public struct EdgeToolsTokenizerLoadingError: Hashable, Sendable, Error {
  public let message: String

  public static func noCompatibleTokenizer(
    in directoryURL: URL,
    hasSentencePieceModel: Bool,
    hasTransformersTokenizer: Bool,
    failures: [String] = []
  ) -> Self {
    var details = [String]()

    if hasSentencePieceModel {
      details.append("tokenizer.model is not a supported SentencePiece BPE model.")
    } else {
      details.append("tokenizer.model was not found.")
    }

    #if Transformers
      if hasTransformersTokenizer {
        details.append("tokenizer.json is not a supported Transformers tokenizer.")
      } else {
        details.append("tokenizer.json was not found.")
      }
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

    details.append(contentsOf: failures)
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
  ) async throws -> sending any EdgeToolsTokenizer {
    let tokenizer = try await AutoTokenizer.from(modelFolder: directoryURL)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
      throw EdgeToolsTokenizerLoadingError.unsupportedTransformersTokenizer(at: tokenizerURL)
    }
    return tokenizer
  }
#endif
