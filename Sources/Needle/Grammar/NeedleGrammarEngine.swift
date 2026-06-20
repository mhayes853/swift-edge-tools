public protocol NeedleGrammarEngine<Matcher> {
  associatedtype Matcher

  nonisolated(nonsending) func compile(tools: [NeedleToolDefinition]) async throws -> Matcher
}
