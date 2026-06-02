// MARK: - NeedleEngine

public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine

  func prefill(prompt: String, tools: [NeedleToolDefinition]) throws -> NeedlePrefillMetrics

  func generate(
    prompt: String,
    tools: [NeedleToolDefinition],
    matcher: GrammarEngine.Matcher,
    onToken: (NeedleToken) -> Void
  ) throws -> NeedleEngineGeneration
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
