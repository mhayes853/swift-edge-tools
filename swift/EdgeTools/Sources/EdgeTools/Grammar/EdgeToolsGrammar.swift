// MARK: - EdgeToolsGrammarMatcher

public protocol EdgeToolsGrammarMatcher: ~Copyable {
  var isTerminated: Bool { get }

  mutating func grammarBitmask() -> GrammarBitmask

  @discardableResult
  mutating func accept(tokenId: EdgeToolsToken.ID) -> Bool
}

// MARK: - EdgeToolsGrammarEngine

public protocol EdgeToolsGrammarEngine: Sendable {
  associatedtype Grammar: Sendable & ~Copyable
  associatedtype Matcher: EdgeToolsGrammarMatcher & ~Copyable

  func matcher(
    for grammar: borrowing Grammar,
    stopTokenIds: Set<EdgeToolsToken.ID>
  ) throws -> Matcher
}

// MARK: - EdgeToolsGenerationConstraint

public protocol EdgeToolsGenerationConstraint: Sendable {
  associatedtype Grammar: Sendable & ~Copyable
  associatedtype Context

  var toolCallRange: GrammarToolCallRange? { get }

  func grammar(toolCallGrammar: consuming Grammar?, context: Context) throws -> Grammar
}

// MARK: - EdgeToolsConstrainedGenerateParameters

public protocol EdgeToolsConstrainedGenerateParameters: EdgeToolsEngineGenerateParameters {
  associatedtype Constraint: EdgeToolsGenerationConstraint
  var constraint: Constraint { get }
}
