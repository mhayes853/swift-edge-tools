public struct EdgeToolsReasoningEffort: RawRepresentable, Hashable, Sendable {
  public static let `default` = Self(rawValue: "default")
  public static let none = Self(rawValue: "none")
  public static let minimal = Self(rawValue: "minimal")
  public static let low = Self(rawValue: "low")
  public static let medium = Self(rawValue: "medium")
  public static let high = Self(rawValue: "high")

  public var rawValue: String

  public var isEnabled: Bool {
    self != .none
  }

  public init(rawValue: String) {
    self.rawValue = rawValue
  }
}
