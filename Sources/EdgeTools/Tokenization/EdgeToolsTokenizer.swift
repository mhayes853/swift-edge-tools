import SystemPackage

#if Foundation
  import Foundation
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

// MARK: - EdgeToolsXGRTokenizer

#if XGrammar
  public protocol EdgeToolsXGRTokenizer: EdgeToolsTokenizer {
    func tokenizerInfo(modelVocabularySize: Int?) throws -> XGRTokenizerInfo
  }
#endif

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

  throw EdgeToolsError.noCompatibleTokenizer(
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

#if Transformers
  private func loadTransformersTokenizer(
    from directoryPath: FilePath,
    tokenizerPath: FilePath
  ) async throws -> sending any EdgeToolsTokenizer {
    let directoryURL = URL(fileURLWithPath: directoryPath.string, isDirectory: true)
    let tokenizer = try await AutoTokenizer.from(modelFolder: directoryURL)
    guard let tokenizer = tokenizer as? PreTrainedTokenizer else {
      throw EdgeToolsError(
        code: .unsupportedTransformersTokenizer,
        message: "swift-transformers created an unsupported tokenizer type from \(tokenizerPath)."
      )
    }
    let backendJSON = try loadHuggingFaceBackendJSON(from: tokenizerPath)
    return EdgeToolsPreTrainedTokenizer(
      tokenizer: tokenizer,
      backendJSON: backendJSON
    )
  }
#endif
