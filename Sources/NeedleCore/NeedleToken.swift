public struct NeedleToken: Hashable, Sendable, Identifiable {
  public let id: UInt32
  public let stringValue: String

  public init(id: UInt32, stringValue: String) {
    self.id = id
    self.stringValue = stringValue
  }
}
