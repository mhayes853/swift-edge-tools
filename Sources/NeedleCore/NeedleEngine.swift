// MARK: - NeedleEngine

public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine

  var grammarEngine: GrammarEngine { get }

  func prefill(prompt: NeedlePrompt) throws -> NeedlePrefillMetrics

  func generate(
    prompt: NeedlePrompt,
    matcher: GrammarEngine.Matcher,
    onToken: (NeedleToken) -> Void
  ) throws -> NeedleEngineGeneration

  func reset()
}

// MARK: - NeedleEngineGeneration

public struct NeedleEngineGeneration: Hashable, Sendable {
  public let prefillMetrics: NeedlePrefillMetrics
  public let decodeMetrics: NeedleDecodeMetrics
  public let tokens: [NeedleToken]

  public init(
    prefillMetrics: NeedlePrefillMetrics,
    decodeMetrics: NeedleDecodeMetrics,
    tokens: [NeedleToken]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.tokens = tokens
  }
}
