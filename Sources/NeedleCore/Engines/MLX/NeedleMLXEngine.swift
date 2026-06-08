#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation

  // MARK: - NeedleMLX

  public final class NeedleMLXEngine<GrammarEngine: NeedleGrammarEngine>
  where GrammarEngine.Matcher: NeedleGrammarMatcher {
    public init(from url: URL) throws {
    }

    public func prefill(prompt: NeedlePrompt) throws -> NeedlePrefillMetrics {
      fatalError()
    }

    public func generate(
      prompt: NeedlePrompt,
      matcher: GrammarEngine.Matcher,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      fatalError()
    }

    public func reset() {}
  }

  // MARK: - XGrammar

  extension NeedleMLXEngine: NeedleEngine where GrammarEngine == NeedleXGrammarEngine {
    public var grammarEngine: GrammarEngine {
      fatalError()
    }
  }
#endif
