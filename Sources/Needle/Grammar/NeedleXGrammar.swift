#if SwiftNeedleXGrammar

  // MARK: - NeedleXGrammarEngine

  public struct NeedleXGrammarEngine: NeedleGrammarEngine {
    public init() {
    }

    public func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
      fatalError()
    }
  }

  // MARK: - Matcher

  extension NeedleXGrammarEngine {
    public struct Matcher: NeedleGrammarMatcher {
      public func bitmask() -> NeedleGrammarBitmask {
        NeedleGrammarBitmask()
      }

      public mutating func accept(tokenId: Int) {
      }
    }
  }
#endif
