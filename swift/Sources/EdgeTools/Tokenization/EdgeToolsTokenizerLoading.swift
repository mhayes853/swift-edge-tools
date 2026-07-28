#if System
  import SystemPackage
#endif

#if Foundation
  package import _EdgeToolsFoundation
#endif

#if Transformers
  import Tokenizers
#endif

// MARK: - Tokenizer Loading

#if System
  package func loadEdgeToolsTokenizer(
    from directoryPath: FilePath
  ) async throws -> sending any EdgeToolsTokenizer {
    let sentencePiecePath = directoryPath.appending("tokenizer.model")
    let transformersPath = directoryPath.appending("tokenizer.json")
    let hasSentencePieceModel = fileExists(at: sentencePiecePath)
    let hasTransformersTokenizer = fileExists(at: transformersPath)
    var failures = [String]()

    if hasSentencePieceModel {
      do {
        return try NeedleSPTokenizer(modelPath: sentencePiecePath)
      } catch {
        failures.append("tokenizer.model could not be loaded: \(error)")
      }
    }

    #if Transformers && Foundation
      if hasTransformersTokenizer {
        do {
          return try await loadTransformersTokenizer(
            from: URL(filePath: directoryPath.string, directoryHint: .isDirectory),
            tokenizerURL: URL(filePath: transformersPath.string)
          )
        } catch {
          failures.append("tokenizer.json could not be loaded: \(error)")
        }
      }
    #endif

    throw EdgeToolsError.noCompatibleTokenizer(
      in: directoryPath.string,
      hasSentencePieceModel: hasSentencePieceModel,
      hasTransformersTokenizer: hasTransformersTokenizer,
      failures: failures
    )
  }
#endif

#if Foundation
  package func loadEdgeToolsTokenizer(
    from directoryURL: URL
  ) async throws -> sending any EdgeToolsTokenizer {
    let sentencePieceURL = directoryURL.appending(path: "tokenizer.model")
    let transformersURL = directoryURL.appending(path: "tokenizer.json")
    let hasSentencePieceModel = fileExists(at: sentencePieceURL)
    let hasTransformersTokenizer = fileExists(at: transformersURL)
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

    throw EdgeToolsError.noCompatibleTokenizer(
      in: directoryURL.path(),
      hasSentencePieceModel: hasSentencePieceModel,
      hasTransformersTokenizer: hasTransformersTokenizer,
      failures: failures
    )
  }
#endif

#if Transformers && Foundation
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
    return EdgeToolsPreTrainedTokenizer(
      tokenizer: tokenizer,
      backendJSON: backendJSON
    )
  }
#endif