#if SwiftNeedleMLX
  import MLX
  import Foundation

  public struct NeedleMLX<GrammarEngine: NeedleGrammarEngine>: NeedleEngine
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
#endif
