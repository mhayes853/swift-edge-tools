import Foundation

public struct NeedlePrompt: Hashable, Sendable {
  public var system: String
  public var user: String
  public var tools: [NeedleToolDefinition]

  public init(system: String, user: String, tools: [NeedleToolDefinition]) {
    self.system = system
    self.user = user
    self.tools = tools
  }

  public func formatted() throws -> String {
    let toolsSchema = try String(
      decoding: JSONEncoder.needleTools.encode(tools.map { $0.normalized() }),
      as: UTF8.self
    )
    return "\(self.system)\n\n\(self.user)<tools>\(toolsSchema)</s>"
  }
}
