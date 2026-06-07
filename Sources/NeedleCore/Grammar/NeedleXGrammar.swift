#if SwiftNeedleXGrammar
  import XGrammar

  // MARK: - NeedleXGrammarEngine

  public struct NeedleXGrammarEngine: NeedleGrammarEngine {
    public let compiler: Grammar.Compiler

    public init(compiler: Grammar.Compiler) {
      self.compiler = compiler
    }

    public func compile(tools: [NeedleToolDefinition]) async throws -> Matcher {
      fatalError()
    }
  }

  // MARK: - Matcher

  extension NeedleXGrammarEngine {
    public struct Matcher: NeedleGrammarMatcher {
      fileprivate let matcher: Grammar.Matcher

      public func bitmask() -> NeedleGrammarBitmask {
        NeedleGrammarBitmask()
      }

      public mutating func accept(tokenId: Int) {
        self.matcher.accept(Int32(tokenId))
      }
    }
  }
#endif
