// MARK: - EdgeToolsMetadataKey

public struct EdgeToolsMetadataKey: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: StringLiteralType) {
    self.init(rawValue: value)
  }
}

// MARK: - EdgeToolsMetadata

public typealias EdgeToolsMetadata = [EdgeToolsMetadataKey: any Sendable]
