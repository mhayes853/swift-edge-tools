// MARK: - EdgeToolsPrompt

public struct EdgeToolsPrompt: Hashable, Sendable {
  public var system: String
  public var user: String
  public var tools: [EdgeToolDefinition]

  public init(system: String, user: String, tools: [EdgeToolDefinition]) {
    self.system = system
    self.user = user
    self.tools = tools
  }
}
