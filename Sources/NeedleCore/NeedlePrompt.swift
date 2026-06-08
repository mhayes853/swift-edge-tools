import Foundation

public struct NeedlePrompt: Hashable, Sendable {
  private static let jsonEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return encoder
  }()

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
      decoding: Self.jsonEncoder.encode(tools.map { $0.normalized() }),
      as: UTF8.self
    )
    return "\(self.system)\n\n\(self.user)<tools>\(toolsSchema)</s>"
  }
}
