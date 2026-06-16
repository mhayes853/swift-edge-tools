#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLMCommon
  import Foundation

  // MARK: - NeedleMLX

  public final class NeedleMLXEngine: NeedleEngine {
    public typealias GrammarEngine = NeedleXGrammarEngine

    public let grammarEngine: NeedleXGrammarEngine

    public init(
      from url: URL,
      grammarConfiguration: NeedleXGrammarEngine.Configuration =
        NeedleXGrammarEngine.Configuration()
    ) throws {
      self.grammarEngine = NeedleXGrammarEngine(
        encodedVocab: ["</s>"],
        eosTokenId: 0,
        configuration: grammarConfiguration
      )
    }

    public func prefill(prompt: NeedlePrompt) throws -> NeedlePrefillMetrics {
      NeedlePrefillMetrics(tokens: 0, duration: .zero, ramUsageBytes: 0)
    }

    public func generate(
      prompt: NeedlePrompt,
      matcher: GrammarEngine.Matcher,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      NeedleEngineGeneration(
        prefillMetrics: NeedlePrefillMetrics(tokens: 0, duration: .zero, ramUsageBytes: 0),
        decodeMetrics: NeedleDecodeMetrics(
          tokens: 0,
          duration: .zero,
          durationToFirstToken: .zero,
          ramUsageBytes: 0
        ),
        response: ""
      )
    }

    public func reset() {}
  }
#endif
