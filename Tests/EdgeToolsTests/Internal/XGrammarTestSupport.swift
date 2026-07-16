#if XGrammar && Sentencepiece
  import CustomDump
  import EdgeTools
  import Testing

  func makeTestTokenizer() throws -> EdgeToolsSPTokenizer {
    try EdgeToolsSPTokenizer(modelURL: .testTokenizerModel)
  }

  func requiredTestEOSToken(
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) throws -> EdgeToolsToken.ID {
    guard let eosToken = tokenizer.eosTokenId else {
      throw XGrammarError(message: "The test tokenizer must provide an EOS token.")
    }
    return eosToken
  }

  extension XGrammarCompiler {
    func makeMatcher(_ grammar: borrowing XGrammarGrammar) throws -> XGrammarMatcher {
      let compiledGrammar = try self.compile(grammar)
      return try XGrammarMatcher(compiledGrammar: compiledGrammar)
    }
  }

  func makeGenericXGrammarCompiler(
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) throws -> XGrammarCompiler {
    let vocabulary = tokenizer.convertIdsToTokens(Array(0..<8192))
    guard let eosToken = tokenizer.eosTokenId, vocabulary.allSatisfy({ $0 != nil }) else {
      throw XGrammarError(
        message: "The test tokenizer must provide an EOS token and full vocabulary."
      )
    }
    let tokenizerInfo = try XGrammarTokenizerInfo(
      encodedVocabulary: vocabulary.compactMap { $0 },
      vocabularyType: .byteFallback,
      vocabularySize: vocabulary.count,
      stopTokenIDs: [eosToken],
      addPrefixSpace: true
    )
    return try XGrammarCompiler(tokenizerInfo: tokenizerInfo)
  }

  func encodedGrammarText(
    _ text: String,
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
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
    matcher: borrowing XGrammarMatcher,
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable,
    eosToken: EdgeToolsToken.ID
  ) {
    if let rejected = firstRejectedGrammarToken(in: text, matcher: matcher, tokenizer: tokenizer) {
      Issue.record(
        "Rejected token \(rejected.tokenId) '\(rejected.token)' at index \(rejected.index) for prefix: \(rejected.prefix)"
      )
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), true)
  }

  func assertGrammarRejects(
    _ text: String,
    matcher: borrowing XGrammarMatcher,
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable,
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
    matcher: borrowing XGrammarMatcher,
    tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
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
