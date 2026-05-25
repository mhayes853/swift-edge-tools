public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine

  func prefill(prompt: String, tools: [NeedleToolDefinition]) throws
  func generate(
    prompt: String,
    tools: [NeedleToolDefinition],
    matcher: GrammarEngine.Matcher,
    onToken: (NeedleToken) -> Void
  ) throws
}
