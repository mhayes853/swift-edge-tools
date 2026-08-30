import EdgeToolsCore

// MARK: - EdgeToolsGrammarMatcher

public protocol EdgeToolsGrammarMatcher: ~Copyable {
  var isTerminated: Bool { get }

  /// Returns the current vocabulary constraint, or `nil` when every token is accepted.
  mutating func grammarBitmask() -> GrammarBitmask?

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

// MARK: - EdgeToolsSchemaGenerationConstraint

public protocol EdgeToolsSchemaGenerationConstraint: EdgeToolsGenerationConstraint {
  static func schema(_ schema: EdgeToolsGenerationSchema) -> Self
}


extension EdgeToolsSchemaGenerationConstraint {
  public static func schema(_ type: (some EdgeToolsGenerable).Type) -> Self {
    .schema(type.edgeToolsGenerationSchema)
  }
}

// MARK: - EdgeToolsTurnGenerationConstraint

public protocol EdgeToolsTurnGenerationConstraint: EdgeToolsGenerationConstraint {
  static func toolCallsOrResponse<Response: EdgeToolsGenerable>(
    _ response: Response.Type,
    toolCallRange: GrammarToolCallRange
  ) -> Self
}

// MARK: - EdgeToolsConstrainedGenerateParameters

public protocol EdgeToolsConstrainedGenerateParameters: EdgeToolsEngineGenerateParameters {
  associatedtype Constraint: EdgeToolsGenerationConstraint
  var constraint: Constraint { get set }
}
