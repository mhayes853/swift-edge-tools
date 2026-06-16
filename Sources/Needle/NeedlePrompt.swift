import Foundation

// MARK: - NeedlePrefillablePrompt

public struct NeedlePrefillablePrompt: Hashable, Sendable {
  public var system: String
  public var user: String

  public init(system: String, user: String) {
    self.system = system
    self.user = user
  }

  public init(prompt: NeedlePrompt) {
    self.init(system: prompt.system, user: prompt.user)
  }

  public func formatted() -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    return "\(self.system)\(separator)\(self.user)"
  }
}

// MARK: - NeedlePrompt

public struct NeedlePrompt: Hashable, Sendable {
  public var system: String
  public var user: String
  public var tools: [NeedleToolDefinition]

  public init(system: String, user: String, tools: [NeedleToolDefinition]) {
    self.system = system
    self.user = user
    self.tools = tools
  }

  public init(prefillable: NeedlePrefillablePrompt, tools: [NeedleToolDefinition]) {
    self.init(system: prefillable.system, user: prefillable.user, tools: tools)
  }

  public func formatted() throws -> String {
    let toolsSchema = try String(
      decoding: JSONEncoder.needleTools.encode(tools.map { $0.normalized() }),
      as: UTF8.self
    )
    return "\(NeedlePrefillablePrompt(prompt: self).formatted())<tools>\(toolsSchema)</s>"
  }
}
