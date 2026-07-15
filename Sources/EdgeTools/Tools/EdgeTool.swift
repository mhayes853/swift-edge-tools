// MARK: - EdgeTool

public protocol EdgeTool<Input, Output>: Sendable {
  associatedtype Input: ConvertibleFromEdgeToolsValue & Sendable
  associatedtype Output: Sendable

  var name: String { get }
  var description: String { get }
  var arguments: EdgeToolsGenerationSchema { get }
  var includesSchemaInInstructions: Bool { get }

  func invoke(input: Input) async throws -> Output
}

extension EdgeTool where Input: EdgeToolsGenerable {
  public var arguments: EdgeToolsGenerationSchema {
    Input.edgeToolsGenerationSchema
  }
}

extension EdgeTool {
  public var includesSchemaInInstructions: Bool { true }

  public var definition: EdgeToolDefinition {
    EdgeToolDefinition(
      name: self.name,
      description: self.description,
      arguments: self.arguments,
      includesSchemaInInstructions: self.includesSchemaInInstructions
    )
  }
}

// MARK: - EdgeToolDefinition

public struct EdgeToolDefinition: Hashable, Sendable, Codable {
  public var name: String
  public var description: String
  public var arguments: EdgeToolsGenerationSchema
  public var includesSchemaInInstructions: Bool

  public init(
    name: String,
    description: String,
    arguments: EdgeToolsGenerationSchema,
    includesSchemaInInstructions: Bool = true
  ) {
    self.name = name
    self.description = description
    self.arguments = arguments
    self.includesSchemaInInstructions = includesSchemaInInstructions
  }
}
