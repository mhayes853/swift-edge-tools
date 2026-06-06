import Needle

struct AlwaysGrammarEngine: NeedleGrammarEngine {
  struct Matcher: NeedleGrammarMatcher {
    func bitmask() -> NeedleGrammarBitmask {
      var mask = NeedleGrammarBitmask()
      for i in 0..<mask.count {
        mask[i] = true
      }
      return mask
    }

    func accept(token: NeedleCore.NeedleToken) {
    }
  }

  func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
    Matcher()
  }
}
