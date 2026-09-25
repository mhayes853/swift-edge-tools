import EdgeToolsCore

// MARK: - EdgeToolsGrammarGuidance

/// How a grammar guides the decoding of the next token.
public struct EdgeToolsGrammarGuidance: Hashable, Sendable {
  /// The tokens the grammar accepts next, or `nil` when every token is accepted.
  public var bitmask: GrammarBitmask?

  /// The sampling parameters the grammar prefers for the next token.
  ///
  /// Engines apply these on top of their default sampling, and parameters requested for a
  /// generation take precedence over them.
  public var sampling: EdgeToolsFusedSamplingParameters

  public init(
    bitmask: GrammarBitmask? = nil,
    sampling: EdgeToolsFusedSamplingParameters = EdgeToolsFusedSamplingParameters()
  ) {
    self.bitmask = bitmask
    self.sampling = sampling
  }
}

// MARK: - EdgeToolsGrammarMatcher

public protocol EdgeToolsGrammarMatcher: ~Copyable {
  var isTerminated: Bool { get }

  /// Returns how the grammar guides the next token.
  mutating func nextTokenGuidance() -> EdgeToolsGrammarGuidance

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
