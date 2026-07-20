import SystemPackage

#if Foundation
  #if canImport(FoundationEssentials)
    import FoundationEssentials
  #else
    import Foundation
  #endif
#endif

#if Transformers
  import Tokenizers
#endif

// MARK: - EdgeToolsTokenizer

public protocol EdgeToolsTokenizer: Sendable {
  var unknownTokenId: EdgeToolsToken.ID? { borrowing get }
  var bosTokenId: EdgeToolsToken.ID? { borrowing get }
  var eosTokenId: EdgeToolsToken.ID? { borrowing get }

  func encode(text: String) -> [EdgeToolsToken.ID]
  func decode(tokens: [EdgeToolsToken.ID]) -> String
  func convertTokensToIds(_ tokens: [String]) -> [EdgeToolsToken.ID?]
  func convertIdsToTokens(_ ids: [EdgeToolsToken.ID]) -> [String?]
}

extension EdgeToolsTokenizer {
  public func tokenize(text: String) -> [String] {
    self.encode(text: text).compactMap { self.convertIdToToken($0) }
  }

  public func convertTokenToId(_ token: String) -> EdgeToolsToken.ID? {
    self.convertTokensToIds([token])[0]
  }

  public func convertIdToToken(_ id: EdgeToolsToken.ID) -> String? {
    self.convertIdsToTokens([id])[0]
  }

  public var unknownToken: String? {
    guard let tokenID = self.unknownTokenId else { return nil }
    return self.convertIdToToken(tokenID)
  }

  public var bosToken: String? {
    guard let tokenID = self.bosTokenId else { return nil }
    return self.convertIdToToken(tokenID)
  }

  public var eosToken: String? {
    guard let tokenID = self.eosTokenId else { return nil }
    return self.convertIdToToken(tokenID)
  }
}

// MARK: - Tokenizer Loading

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

  #if Transformers
    if hasTransformersTokenizer {
      do {
        return try await loadTransformersTokenizer(
          from: directoryPath,
          tokenizerPath: transformersPath
        )
      } catch {
        failures.append("tokenizer.json could not be loaded: \(error)")
      }
    }
  #endif

  throw EdgeToolsTokenizerLoadingError.noCompatibleTokenizer(
    in: directoryPath,
    hasSentencePieceModel: hasSentencePieceModel,
    hasTransformersTokenizer: hasTransformersTokenizer,
    failures: failures
  )
}

#if Foundation
  package func loadEdgeToolsTokenizer(
    from directoryURL: URL
  ) async throws -> sending any EdgeToolsTokenizer {
    try await loadEdgeToolsTokenizer(from: FilePath(directoryURL.path()))
  }
#endif

public struct EdgeToolsTokenizerLoadingError: Hashable, Sendable, Error {
  public let message: String

  public static func noCompatibleTokenizer(
    in directoryPath: FilePath,
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
        "No compatible tokenizer was found in \(directoryPath). \(details.joined(separator: " "))"
    )
  }

  public static func unsupportedTransformersTokenizer(at tokenizerPath: FilePath) -> Self {
    Self(
      message:
        "swift-transformers created an unsupported tokenizer type from \(tokenizerPath)."
    )
  }
}

#if Transformers
  private func loadTransformersTokenizer(
    from directoryPath: FilePath,
    tokenizerPath: FilePath
  ) async throws -> sending any EdgeToolsTokenizer {
    let directoryURL = URL(fileURLWithPath: directoryPath.string, isDirectory: true)
    let tokenizer = try await AutoTokenizer.from(modelFolder: directoryURL)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
      throw EdgeToolsTokenizerLoadingError.unsupportedTransformersTokenizer(at: tokenizerPath)
    }
    let backendJSON = try loadHuggingFaceBackendJSON(from: tokenizerPath)
    return EdgeToolsPreTrainedTokenizer(
      tokenizer: tokenizer,
      backendJSON: backendJSON
    )
  }
#endif
