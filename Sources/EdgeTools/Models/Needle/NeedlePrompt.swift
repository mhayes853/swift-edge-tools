import Foundation

// MARK: - NeedlePrompt

public struct NeedlePrompt: Sendable {
  public var system: String
  public var user: String

  public init(system: String, user: String) {
    self.system = system
    self.user = user
  }
}

// MARK: - Formatting

extension NeedlePrompt {
  public func formatted(tools: [EdgeToolDefinition]) throws -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema =
      try tools
      .filter(\.includesSchemaInInstructions)
      .map { $0.needleNormalized() }
      .needlePromptEncoded()
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }

  public func tokenized(
    tools: [EdgeToolDefinition],
    using tokenizer: borrowing some EdgeToolsTokenizer
  ) throws -> [EdgeToolsToken] {
    let tokenIds = tokenizer.encode(text: try self.formatted(tools: tools))
    let tokens = tokenizer.convertIdsToTokens(tokenIds)
    return zip(tokenIds, tokens)
      .compactMap { (tokenId, token) in
        token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
      }
  }
}

// MARK: - EdgeToolDefinition

extension EdgeToolDefinition {
  public func needleNormalized() -> Self {
    var definition = self
    definition.name = self.name.snakeCased()
    return definition
  }

  fileprivate func needlePromptEncoded() throws -> String {
    let name = String(decoding: try JSONEncoder().encode(self.name), as: UTF8.self)
    let description = String(decoding: try JSONEncoder().encode(self.description), as: UTF8.self)
    let arguments = self.arguments.orderedKeyEncoded()
    return #"{"name":\#(name),"description":\#(description),"arguments":\#(arguments)}"#
  }
}

extension Sequence where Element == EdgeToolDefinition {
  fileprivate func needlePromptEncoded() throws -> String {
    let definitions = try self.map { try $0.needlePromptEncoded() }
    return "[\(definitions.joined(separator: ","))]"
  }
}
