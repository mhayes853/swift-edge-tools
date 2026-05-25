public protocol NeedleGrammarEngine<Matcher> {
  associatedtype Matcher

  func compile(tools: [NeedleToolDefinition]) async throws -> Matcher
}
