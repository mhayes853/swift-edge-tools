// MARK: - NeedleEngine

public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine

  func prefill(prompt: String, tools: [NeedleToolDefinition]) throws -> NeedlePrefillMetrics

  func generate(
    prompt: String,
    tools: [NeedleToolDefinition],
    matcher: GrammarEngine.Matcher,
    onToken: (NeedleToken) -> Void
  ) throws -> NeedleEngineCompletion
}

// MARK: - NeedleEngineCompletion

public struct NeedleEngineCompletion: Hashable, Sendable {
  public let prefillMetrics: NeedlePrefillMetrics
  public let decodeMetrics: NeedleDecodeMetrics
  public let toolCalls: [NeedleRawToolCall]

  public init(
    prefillMetrics: NeedlePrefillMetrics,
    decodeMetrics: NeedleDecodeMetrics,
    toolCalls: [NeedleRawToolCall]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.toolCalls = toolCalls
  }
}
