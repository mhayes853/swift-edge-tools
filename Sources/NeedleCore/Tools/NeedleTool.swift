// MARK: - NeedleTool

public protocol NeedleTool<Input, Output> {
  associatedtype Input: ConvertibleFromNeedleValue
  associatedtype Output

  var name: String { get }
  var description: String { get }
  var arguments: NeedleGenerationSchema { get }

  func invoke(input: sending Input) async throws -> sending Output
}

extension NeedleTool where Input: NeedleGenerable {
  public var arguments: NeedleGenerationSchema {
    Input.needleGenerationSchema
  }
}

extension NeedleTool {
  public var definition: NeedleToolDefinition {
    NeedleToolDefinition(
      name: self.name,
      description: self.description,
      arguments: self.arguments
    )
  }
}

// MARK: - NeedleToolDefinition

public struct NeedleToolDefinition: Hashable, Sendable, Codable {
  public var name: String
  public var description: String
  public var arguments: NeedleGenerationSchema

  public init(name: String, description: String, arguments: NeedleGenerationSchema) {
    self.name = name
    self.description = description
    self.arguments = arguments
  }
}
