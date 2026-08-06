// MARK: - EdgeToolsGrammarMatcher

public protocol EdgeToolsGrammarMatcher: ~Copyable {
  var isTerminated: Bool { get }

  mutating func bitmask() -> GrammarBitmask

  @discardableResult
  mutating func accept(tokenId: EdgeToolsToken.ID) -> Bool
}

// MARK: - EdgeToolsGrammarCompiler

public protocol EdgeToolsGrammarCompiler: ~Copyable {
  associatedtype Grammar: Sendable
  associatedtype Context = Void
  associatedtype Matcher: EdgeToolsGrammarMatcher, ~Copyable

  mutating func matcher(
    for grammar: Grammar,
    context: borrowing Context
  ) throws -> Matcher
}

// MARK: - EdgeToolsGenerationConstraint

public protocol EdgeToolsGenerationConstraint: Sendable {
  associatedtype Grammar: Sendable
  associatedtype Context

  var toolCallRange: GrammarToolCallRange? { get }

  func grammar(toolCallGrammar: Grammar?, context: Context) throws -> Grammar
}

// MARK: - EdgeToolsConstrainedGenerateParameters

public protocol EdgeToolsConstrainedGenerateParameters: EdgeToolsEngineGenerateParameters {
  associatedtype Constraint: EdgeToolsGenerationConstraint
  var constraint: Constraint { get }
}
