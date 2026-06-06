public struct NeedleToken: Hashable, Sendable, Identifiable {
  public let id: Int
  public let stringValue: String

  public init(id: Int, stringValue: String) {
    self.id = id
    self.stringValue = stringValue
  }
}
