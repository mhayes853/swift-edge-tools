#if XGrammar
  import CustomDump
  import EdgeTools
  import Testing

  #if System
    import SystemPackage
  #endif

  func testTokenizer() throws -> NeedleSPTokenizer {
    #if Foundation
      try NeedleSPTokenizer(modelURL: .testTokenizerModel)
    #elseif System
      try NeedleSPTokenizer(modelPath: .testTokenizerModel)
    #else
      throw XGRError(
        code: .invalidNeedleTokenizer,
        message: "Test tokenizer loading requires the Foundation or System trait."
      )
    #endif
  }

  func requiredTestEOSToken(
    tokenizer: some EdgeToolsXGRTokenizer
  ) throws -> EdgeToolsToken.ID {
    guard let eosToken = tokenizer.eosTokenId else {
      throw XGRError(
        code: .invalidNeedleTokenizer,
        message: "The test tokenizer must provide an EOS token."
      )
    }
    return eosToken
  }

  extension XGRCompiler {
    func makeMatcher(_ grammar: borrowing XGRGrammar) throws -> XGRMatcher {
      let compiledGrammar = try self.compile(grammar)
      return try XGRMatcher(compiledGrammar: compiledGrammar)
    }
  }

  func makeGenericXGRCompiler(
    tokenizer: some EdgeToolsXGRTokenizer
  ) throws -> XGRCompiler {
    let vocabulary = tokenizer.convertIdsToTokens(Array(0..<8192))
    guard let eosToken = tokenizer.eosTokenId, vocabulary.allSatisfy({ $0 != nil }) else {
      throw XGRError(
        code: .invalidNeedleTokenizer,
        message: "The test tokenizer must provide an EOS token and full vocabulary."
      )
    }
    let tokenizerInfo = try XGRTokenizerInfo(
      encodedVocabulary: vocabulary.compactMap { $0 },
      vocabularyType: .byteFallback,
      vocabularySize: vocabulary.count,
      stopTokenIDs: [eosToken],
      addPrefixSpace: true
    )
    return try XGRCompiler(tokenizerInfo: tokenizerInfo)
  }

  func encodedGrammarText(
    _ text: String,
    tokenizer: some EdgeToolsXGRTokenizer
  ) -> [EdgeToolsToken.ID] {
    let tokenIds = tokenizer.encode(text: text)
    guard let firstTokenId = tokenIds.first else { return tokenIds }
    let firstToken = tokenizer.convertIdToToken(firstTokenId) ?? ""
    if firstToken.hasPrefix("▁") {
      return Array(tokenIds.dropFirst())
    }
    return tokenIds
  }

  func assertGrammarAccepts(
    _ text: String,
    matcher: borrowing XGRMatcher,
    tokenizer: some EdgeToolsXGRTokenizer,
    eosToken: EdgeToolsToken.ID
  ) {
    if let rejected = firstRejectedGrammarToken(in: text, matcher: matcher, tokenizer: tokenizer) {
      let message = [
        "Rejected token \(rejected.tokenId) '\(rejected.token)' at index \(rejected.index)",
        " for prefix: \(rejected.prefix)"
      ]
      .joined()
      Issue.record("\(message)")
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), true)
  }

  func assertGrammarRejects(
    _ text: String,
    matcher: borrowing XGRMatcher,
    tokenizer: some EdgeToolsXGRTokenizer,
    eosToken: EdgeToolsToken.ID
  ) {
    guard firstRejectedGrammarToken(in: text, matcher: matcher, tokenizer: tokenizer) == nil else {
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), false)
  }

  private struct RejectedGrammarToken: Hashable, Sendable {
    let index: Int
    let tokenId: EdgeToolsToken.ID
    let token: String
    let prefix: String
  }

  private func firstRejectedGrammarToken(
    in text: String,
    matcher: borrowing XGRMatcher,
    tokenizer: some EdgeToolsXGRTokenizer
  ) -> RejectedGrammarToken? {
    let tokenIds = encodedGrammarText(text, tokenizer: tokenizer)
    for (index, tokenId) in tokenIds.enumerated() {
      guard !matcher.accept(tokenId: tokenId) else { continue }
      let token = tokenizer.convertIdToToken(tokenId) ?? ""
      let prefix = tokenizer.decode(tokens: Array(tokenIds.prefix(index + 1)))
      return RejectedGrammarToken(index: index, tokenId: tokenId, token: token, prefix: prefix)
    }
    return nil
  }
#endif
