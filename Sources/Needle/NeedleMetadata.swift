// MARK: - NeedleMetadataKey

public struct NeedleMetadataKey: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: StringLiteralType) {
    self.init(rawValue: value)
  }
}

// MARK: - NeedleMetadata

public typealias NeedleMetadata = [NeedleMetadataKey: any Sendable]
