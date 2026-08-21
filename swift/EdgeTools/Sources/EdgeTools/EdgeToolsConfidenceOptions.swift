public struct EdgeToolsConfidenceOptions: OptionSet, Hashable, Sendable {
  public let rawValue: UInt8

  public init(rawValue: UInt8) {
    self.rawValue = rawValue
  }

  public static let generation = Self(rawValue: 1 << 0)
  public static let perToken = Self(rawValue: 1 << 1)
  public static let probe = Self(rawValue: 1 << 2)
}
