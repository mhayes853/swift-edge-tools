#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation

  // MARK: - NeedleMLX

  public struct NeedleMLXEngine<GrammarEngine: NeedleGrammarEngine>
  where GrammarEngine.Matcher: NeedleGrammarMatcher {
    public init(from url: URL) throws {
    }

    public func prefill(
      prompt: String,
      tools: [NeedleToolDefinition]
    ) throws -> NeedlePrefillMetrics {
      fatalError()
    }

    public func generate(
      prompt: String,
      tools: [NeedleToolDefinition],
      matcher: GrammarEngine.Matcher,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      fatalError()
    }
  }

  // MARK: - XGrammar

  extension NeedleMLXEngine: NeedleEngine where GrammarEngine == NeedleXGrammarEngine {
    public var grammarEngine: GrammarEngine {
      fatalError()
    }
  }
#endif
