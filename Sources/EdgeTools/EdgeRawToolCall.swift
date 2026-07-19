public struct EdgeRawToolCall: Hashable, Sendable, Codable {
  public let name: String
  public let arguments: EdgeToolsValue

  public init(name: String, arguments: EdgeToolsValue) {
    self.name = name
    self.arguments = arguments
  }
}

extension EdgeRawToolCall {
  package init?(jsonValue: EdgeToolsValue) {
    guard case .object(let object) = jsonValue,
      case .string(let name) = object["name"],
      let arguments = object["arguments"]
    else { return nil }
    self.init(name: name, arguments: arguments)
  }
}
