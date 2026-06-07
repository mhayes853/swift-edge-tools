import Needle

// MARK: - AlwaysGrammarEngine

struct AlwaysGrammarEngine: NeedleGrammarEngine {
  struct Matcher: NeedleGrammarMatcher {
    func bitmask() -> NeedleGrammarBitmask {
      var mask = NeedleGrammarBitmask()
      for i in 0..<mask.count {
        mask[i] = true
      }
      return mask
    }

    func accept(tokenId: Int) {
    }
  }

  func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
    Matcher()
  }
}

// MARK: - ConstantGrammarEngine

struct ConstantGrammarEngine: NeedleGrammarEngine {
  struct Matcher: NeedleGrammarMatcher {
    var mask: NeedleGrammarBitmask

    func bitmask() -> NeedleGrammarBitmask {
      self.mask
    }

    func accept(tokenId: Int) {
    }
  }

  var mask: NeedleGrammarBitmask

  func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
    Matcher(mask: self.mask)
  }
}
