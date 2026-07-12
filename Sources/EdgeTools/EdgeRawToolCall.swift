// MARK: - EdgeRawToolCall

public struct EdgeRawToolCall: Hashable, Sendable, Codable {
  public let name: String
  public let arguments: EdgeToolsValue

  public init(name: String, arguments: EdgeToolsValue) {
    self.name = name
    self.arguments = arguments
  }
}
