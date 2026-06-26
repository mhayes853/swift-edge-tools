public struct NeedlePrompt: Hashable, Sendable {
  public var system: String
  public var user: String
  public var tools: [NeedleToolDefinition]

  public init(system: String, user: String, tools: [NeedleToolDefinition]) {
    self.system = system
    self.user = user
    self.tools = tools
  }

  public func formatted() -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema = String(
      decoding: tools.map { $0.normalized() }.needlePromptEncoded(),
      as: UTF8.self
    )
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }
}
