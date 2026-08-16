import CustomDump
import Testing

@testable import EdgeTools

@Suite
struct `StreamingDetokenizer tests` {
  @Test
  func `Streams A Grapheme Cluster Extended By A Later Token Without Duplicating Text`() {
    let tokenizer = ExtendingGraphemeTokenizer()
    var detokenizer = StreamingDetokenizer()

    let first = detokenizer.decode(tokenId: 0, using: tokenizer)
    let second = detokenizer.decode(tokenId: 1, using: tokenizer)

    expectNoDifference(first + second, "e\u{301}")
    expectNoDifference(first, "e")
    expectNoDifference(second.unicodeScalars.map(\.value), [0x301])
  }
}

private struct ExtendingGraphemeTokenizer: EdgeToolsTokenizer {
  let eos: EdgeToolsToken? = nil

  func encode(text: String) -> [EdgeToolsToken] {
    []
  }

  func decode(tokens: [EdgeToolsToken.ID]) -> String {
    switch tokens {
    case [0]: "e"
    case [0, 1]: "e\u{301}"
    default: ""
    }
  }

  func tokens(forIds ids: [EdgeToolsToken.ID]) -> [EdgeToolsToken?] {
    ids.map { _ in nil }
  }

  func tokens(forTexts texts: [String]) -> [EdgeToolsToken?] {
    texts.map { _ in nil }
  }
}
