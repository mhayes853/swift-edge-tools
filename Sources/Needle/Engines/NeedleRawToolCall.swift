// MARK: - NeedleRawToolCall

public struct NeedleRawToolCall: Hashable, Sendable, Codable {
  public let name: String
  public let arguments: NeedleValue

  public init(name: String, arguments: NeedleValue) {
    self.name = name
    self.arguments = arguments
  }
}
