// MARK: - NeedleEngine

public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine

  var grammarEngine: GrammarEngine { get }

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
  public let response: String

  public init(
    prefillMetrics: NeedlePrefillMetrics,
    decodeMetrics: NeedleDecodeMetrics,
    response: String
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.response = response
  }
}
