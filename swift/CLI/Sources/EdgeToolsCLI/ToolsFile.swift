import EdgeTools
import Foundation

// MARK: - ToolsFile

/// Tool definitions loaded from disk, in either OpenAI function-calling format or the framework's
/// own `EdgeToolDefinition` encoding.
public struct ToolsFile: Sendable {
  public var definitions: [EdgeToolDefinition]

  public init(definitions: [EdgeToolDefinition]) {
    self.definitions = definitions
  }
}

// MARK: - Loading

extension ToolsFile {
  public init(contentsOf url: URL) throws {
    do {
      try self.init(data: Data(contentsOf: url))
    } catch let error as EdgeCLIError {
      throw EdgeCLIError("\(url.lastPathComponent): \(error.description)")
    }
  }

  public init(data: Data) throws {
    let decoder = JSONDecoder()
    let entries: [ToolEntry]
    do {
      entries = try decoder.decode([ToolEntry].self, from: data)
    } catch {
      guard let wrapper = try? decoder.decode(ToolsWrapper.self, from: data) else {
        throw EdgeCLIError(
          """
          Expected a JSON array of tools, or an object with a "tools" array. Tools may use either \
          OpenAI function-calling format or {"name", "description", "arguments"}.
          """
        )
      }
      entries = wrapper.tools
    }
    guard !entries.isEmpty else {
      throw EdgeCLIError("No tools defined.")
    }
    self.init(definitions: entries.map(\.definition))
  }
}

// MARK: - ToolEntry

/// Accepts both `{"type": "function", "function": {...}}` and a bare tool object, and within
/// either, both `parameters` (OpenAI) and `arguments` (framework) for the argument schema.
private struct ToolEntry: Decodable {
  let definition: EdgeToolDefinition

  private struct Function: Decodable {
    let name: String
    let description: String?
    let parameters: EdgeToolsGenerationSchema?
    let arguments: EdgeToolsGenerationSchema?
    let includesSchemaInInstructions: Bool?
  }

  private enum CodingKeys: String, CodingKey {
    case function
  }

  init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let function =
      if let nested = try container.decodeIfPresent(Function.self, forKey: .function) {
        nested
      } else {
        try Function(from: decoder)
      }
    self.definition = EdgeToolDefinition(
      name: function.name,
      description: function.description ?? "",
      arguments: function.parameters ?? function.arguments ?? [:],
      includesSchemaInInstructions: function.includesSchemaInInstructions ?? true
    )
  }
}

// MARK: - ToolsWrapper

private struct ToolsWrapper: Decodable {
  let tools: [ToolEntry]
}
