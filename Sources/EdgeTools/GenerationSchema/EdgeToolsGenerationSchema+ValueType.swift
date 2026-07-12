// MARK: - ValueType

extension EdgeToolsGenerationSchema {
  /// A type-identifier for a ``EdgeToolsGenerationSchema`` value.
  public struct ValueType: Hashable, Sendable, OptionSet {
    public var rawValue: UInt8

    public init(rawValue: UInt8) {
      self.rawValue = rawValue
    }

    /// An integer type.
    public static let integer = Self(rawValue: 1 << 0)

    /// A string type.
    public static let string = Self(rawValue: 1 << 1)

    /// A boolean type.
    public static let boolean = Self(rawValue: 1 << 2)

    /// An array type.
    public static let array = Self(rawValue: 1 << 3)

    /// An object type.
    public static let object = Self(rawValue: 1 << 4)

    /// A number type.
    public static let number = Self(rawValue: 1 << 5)

    /// A null type.
    public static let null = Self(rawValue: 1 << 6)

    /// Returns true if this type is compatible with the type of the specified `value`.
    public func isCompatible(with value: EdgeToolsValue) -> Bool {
      self.contains(value.type) || (value.type == .integer && self.contains(.number))
    }

    public var edgeToolsValue: EdgeToolsValue {
      let containedTypes = self.containedTypes
      if containedTypes.count == 1, let type = containedTypes.first {
        return .string(type.canonicalName)
      }
      return .array(containedTypes.map { .string($0.canonicalName) })
    }
  }
}

extension EdgeToolsGenerationSchema.ValueType: ExpressibleByArrayLiteral {
  public init(arrayLiteral elements: Self...) {
    self.init(elements)
  }
}

extension EdgeToolsGenerationSchema.ValueType: Encodable {
  public func encode(to encoder: any Encoder) throws {
    try self.edgeToolsValue.encode(to: encoder)
  }
}

extension EdgeToolsGenerationSchema.ValueType: Decodable {
  public init(from decoder: any Decoder) throws {
    let value = try EdgeToolsValue(from: decoder)
    guard let type = EdgeToolsGenerationSchema.valueType(from: value) else {
      let container = try decoder.singleValueContainer()
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid schema type")
    }
    self = type
  }
}

extension EdgeToolsGenerationSchema.ValueType {
  var canonicalName: String {
    switch self {
    case .integer: "integer"
    case .string: "string"
    case .boolean: "boolean"
    case .array: "array"
    case .object: "object"
    case .number: "number"
    case .null: "null"
    default: "unknown"
    }
  }

  var containedTypes: [Self] {
    let allTypes = [Self.integer, .string, .boolean, .array, .object, .number, .null]
    return allTypes.filter { self.contains($0) }
  }
}
