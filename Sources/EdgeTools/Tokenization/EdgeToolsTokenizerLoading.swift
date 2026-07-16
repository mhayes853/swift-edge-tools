import Foundation

#if Transformers
  import Tokenizers
#endif

package func loadEdgeToolsTokenizer(
  from directoryURL: URL,
  isNeedleModel: Bool
) async throws -> sending any EdgeToolsTokenizer {
  let fileManager = FileManager.default
  let sentencepieceURL = directoryURL.appending(path: "tokenizer.model")
  let transformersURL = directoryURL.appending(path: "tokenizer.json")

  if isNeedleModel, fileManager.fileExists(atPath: sentencepieceURL.path()) {
    return try NeedleSPTokenizer(modelURL: sentencepieceURL)
  }

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
    isNeedleModel: isNeedleModel,
    hasSentencePieceModel: fileManager.fileExists(atPath: sentencepieceURL.path()),
    hasTransformersTokenizer: fileManager.fileExists(atPath: transformersURL.path())
  )
}

public struct EdgeToolsTokenizerLoadingError: Hashable, Sendable, Error {
  public let message: String

  public static func noCompatibleTokenizer(
    in directoryURL: URL,
    isNeedleModel: Bool,
    hasSentencePieceModel: Bool,
    hasTransformersTokenizer: Bool
  ) -> Self {
    var details = [String]()

    if isNeedleModel {
      if hasSentencePieceModel {
        details.append("tokenizer.model is not a supported Needle SentencePiece BPE model.")
      } else {
        details.append("tokenizer.model was not found.")
      }
    } else {
      details.append("Needle tokenizer.model lookup was skipped for a non-Needle model.")
    }

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

    #if !Transformers
      if isNeedleModel {
        details.append(
          "Provide a Needle tokenizer.model or enable Transformers to load tokenizer.json."
        )
      } else {
        details.append("Enable Transformers to load tokenizer.json.")
      }
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
  ) async throws -> sending any EdgeToolsTokenizer {
    let tokenizer = try await AutoTokenizer.from(modelFolder: directoryURL)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
      throw EdgeToolsTokenizerLoadingError.unsupportedTransformersTokenizer(at: tokenizerURL)
    }
    return tokenizer
  }
#endif
