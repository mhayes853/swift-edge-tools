import EdgeTools

#if XGrammar
  import CustomDump
  import Testing
#endif

struct TestPrompt: Hashable, Sendable {
  var system: String
  var user: String

  init(system: String, user: String) {
    self.system = system
    self.user = user
  }
}

struct TestTokenizer: EdgeToolsTokenizer {
  static let vocabularySize = 258

  let eos: EdgeToolsToken? = EdgeToolsToken(id: 1, stringValue: "<eos>")

  func encode(text: String) -> [EdgeToolsToken] {
    self.tokens(forIds: Array(text.utf8).map { Int($0) + 2 }).compactMap { $0 }
  }

  func decode(tokens: [EdgeToolsToken.ID]) -> String {
    String(decoding: tokens.compactMap { token in
      guard (2..<258).contains(token) else { return nil }
      return UInt8(token - 2)
    }, as: UTF8.self)
  }

  func tokens(forIds ids: [EdgeToolsToken.ID]) -> [EdgeToolsToken?] {
    ids.map { id in
      let stringValue: String?
      switch id {
      case 0: stringValue = "<unk>"
      case 1: stringValue = "<eos>"
      case 2..<Self.vocabularySize: stringValue = Self.tokenString(for: id - 2)
      default: stringValue = nil
      }
      return stringValue.map { EdgeToolsToken(id: id, stringValue: $0) }
    }
  }

  func tokens(forTexts texts: [String]) -> [EdgeToolsToken?] {
    texts.map { text in
      guard text.utf8.count == 1, let byte = text.utf8.first else { return nil }
      return EdgeToolsToken(id: Int(byte) + 2, stringValue: text)
    }
  }

  private static func tokenString(for byte: Int) -> String {
    if [9, 10, 13].contains(byte) {
      return String(UnicodeScalar(byte)!)
    }
    guard (32...126).contains(byte) else {
      return "<unused-\(byte)>"
    }
    return String(UnicodeScalar(byte)!)
  }
}

struct TestGenerationParser: EdgeToolsGenerationParser, Sendable {
  private var buffer = ""

  init() {}

  mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.buffer.append(token.stringValue)
    return self.nextParts()
  }

  mutating func finish() -> [EdgeToolsGenerationPart] {
    defer { self.buffer = "" }
    return self.buffer.isEmpty ? [] : [.text(self.buffer)]
  }

  private mutating func nextParts() -> [EdgeToolsGenerationPart] {
    guard let openerRange = self.buffer.range(of: "<tool_call>") else {
      return []
    }
    let leadingText = String(self.buffer[..<openerRange.lowerBound])
    let source = self.buffer[openerRange.upperBound...]
    guard let jsonStart = source.firstIndex(where: { $0 == "[" || $0 == "{" }),
      let jsonEnd = Self.endOfJSONValue(in: source[jsonStart...])
    else {
      return leadingText.isEmpty ? [] : self.consumeText(leadingText)
    }
    let json = String(source[jsonStart...jsonEnd])
    guard let calls = Self.toolCalls(in: json) else {
      return leadingText.isEmpty ? [] : self.consumeText(leadingText)
    }
    var end = source.index(after: jsonEnd)
    if source[end...].hasPrefix("</tool_call>") {
      end = source.index(end, offsetBy: "</tool_call>".count)
    }
    self.buffer.removeSubrange(..<end)
    return (leadingText.isEmpty ? [] : [.text(leadingText)]) + calls.map(EdgeToolsGenerationPart.toolCall)
  }

  private mutating func consumeText(_ text: String) -> [EdgeToolsGenerationPart] {
    self.buffer.removeFirst(text.count)
    return [.text(text)]
  }

  private static func endOfJSONValue(in source: Substring) -> String.Index? {
    var depth = 0
    var isInsideString = false
    var isEscaping = false

    for index in source.indices {
      let character = source[index]
      if isInsideString {
        if isEscaping {
          isEscaping = false
        } else if character == "\\" {
          isEscaping = true
        } else if character == "\"" {
          isInsideString = false
        }
        continue
      }
      if character == "\"" {
        isInsideString = true
      } else if character == "[" || character == "{" {
        depth += 1
      } else if character == "]" || character == "}" {
        depth -= 1
        if depth == 0 {
          return index
        }
      }
    }
    return nil
  }

  private static func toolCalls(in json: String) -> [EdgeRawToolCall]? {
    guard let value = try? EdgeToolsValue(json: json) else { return nil }
    let values: [EdgeToolsValue]
    if case .array(let array) = value {
      values = array
    } else {
      values = [value]
    }
    return values.compactMap(EdgeRawToolCall.init(jsonValue:))
  }

}

func testTokenizer() throws -> TestTokenizer {
  TestTokenizer()
}

#if XGrammar
  extension TestTokenizer: XGRTokenizer {
    func tokenizerInfo(
      modelVocabularySize: Int?,
      extraStopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> XGRTokenizerInfo {
      let stopTokenIDs = [self.eos?.id].compactMap { $0 } + extraStopTokenIds
      return try XGRTokenizerInfo(
        encodedVocabulary: self.tokens(forIds: Array(0..<Self.vocabularySize)).compactMap { $0?.stringValue },
        vocabularyType: .raw,
        vocabularySize: modelVocabularySize ?? Self.vocabularySize,
        stopTokenIDs: stopTokenIDs,
        addPrefixSpace: false
      )
    }
  }

  func requiredTestEOSToken(
    tokenizer: some XGRTokenizer
  ) throws -> EdgeToolsToken.ID {
    guard let eosTokenId = tokenizer.eos?.id else {
      throw XGRError(code: .invalidTokenizerInfo, message: "The test tokenizer must provide an EOS token.")
    }
    return eosTokenId
  }

  extension XGRCompiler {
    func makeMatcher(_ grammar: borrowing XGRGrammar) throws -> XGRMatcher {
      let compiledGrammar = try self.compile(grammar)
      return try XGRMatcher(compiledGrammar: compiledGrammar)
    }
  }

  func makeGenericXGRCompiler(
    tokenizer: some XGRTokenizer
  ) throws -> XGRCompiler {
    try XGRCompiler(
      tokenizerInfo: tokenizer.tokenizerInfo(modelVocabularySize: nil, extraStopTokenIds: [])
    )
  }

  func encodedGrammarText(
    _ text: String,
    tokenizer: some XGRTokenizer
  ) -> [EdgeToolsToken.ID] {
    tokenizer.encode(text: text).map(\.id)
  }

  func assertGrammarAccepts(
    _ text: String,
    matcher: borrowing XGRMatcher,
    tokenizer: some XGRTokenizer,
    eosToken: EdgeToolsToken.ID
  ) {
    if let rejected = firstRejectedGrammarToken(in: text, matcher: matcher, tokenizer: tokenizer) {
      Issue.record("Rejected token \(rejected.tokenId) '\(rejected.token)' at index \(rejected.index) for prefix: \(rejected.prefix)")
      return
    }
    expectNoDifference(matcher.accept(tokenId: eosToken), true)
  }

  func assertGrammarRejects(
    _ text: String,
    matcher: borrowing XGRMatcher,
    tokenizer: some XGRTokenizer,
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
    tokenizer: some XGRTokenizer
  ) -> RejectedGrammarToken? {
    let tokenIds = encodedGrammarText(text, tokenizer: tokenizer)
    for (index, tokenId) in tokenIds.enumerated() {
      guard !matcher.accept(tokenId: tokenId) else { continue }
      let token = tokenizer.token(forId: tokenId)?.stringValue ?? ""
      let prefix = tokenizer.decode(tokens: Array(tokenIds.prefix(index + 1)))
      return RejectedGrammarToken(index: index, tokenId: tokenId, token: token, prefix: prefix)
    }
    return nil
  }
#endif
