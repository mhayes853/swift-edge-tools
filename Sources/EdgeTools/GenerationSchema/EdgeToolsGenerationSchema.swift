import OrderedCollections

// MARK: - EdgeToolsGenerationSchema

/// An enum defining a generation schema.
///
/// A valid generation schema is either an object or a boolean.
public enum EdgeToolsGenerationSchema: Hashable, Sendable {
  /// A schema key.
  public struct Key: RawRepresentable, ExpressibleByStringLiteral, Hashable, Sendable, Codable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public init(stringLiteral value: StringLiteralType) {
      self.init(rawValue: value)
    }
  }

  /// A boolean schema.
  case boolean(Bool)

  /// An object schema.
  case object(OrderedDictionary<Key, EdgeToolsValue>)

  public init(_ object: OrderedDictionary<Key, EdgeToolsValue>) {
    self = .object(object)
  }

  public init(_ schemas: Self...) {
    self.init(schemas)
  }

  public init(_ schemas: some Sequence<Self>) {
    self = schemas.reduce(Self.object([:])) { $0.merging($1) }
  }

  public var objectValue: OrderedDictionary<Key, EdgeToolsValue>? {
    switch self {
    case .boolean: nil
    case .object(let object): object
    }
  }

  public func merging(_ other: Self) -> Self {
    switch (self, other) {
    case (.object(let lhs), .object(let rhs)):
      var merged = lhs
      for (key, value) in rhs {
        merged[key] = value
      }
      return .object(merged)
    case (_, _):
      return other
    }
  }

  public mutating func merge(_ other: Self) {
    self = self.merging(other)
  }

  public func nullable() -> Self {
    guard case .object(var object) = self else {
      return Self(.anyOf([self, .null]))
    }
    guard let typeValue = object[.type], let valueType = Self.valueType(from: typeValue) else {
      return Self(.anyOf([self, .null]))
    }
    object[.type] = valueType.union(.null).edgeToolsValue
    return .object(object)
  }
}

// MARK: - Codable

extension EdgeToolsGenerationSchema: Encodable {
  public func encode(to encoder: any Encoder) throws {
    switch self {
    case .boolean(let value):
      var container = encoder.singleValueContainer()
      try container.encode(value)
    case .object(let object):
      var container = encoder.container(keyedBy: DynamicCodingKey.self)
      for (key, value) in object {
        try container.encode(value, forKey: DynamicCodingKey(key.rawValue))
      }
    }
  }
}

extension EdgeToolsGenerationSchema: Decodable {
  public init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let bool = try? container.decode(Bool.self) {
      self = .boolean(bool)
      return
    }

    let keyedContainer = try decoder.container(keyedBy: DynamicCodingKey.self)
    var object = OrderedDictionary<Key, EdgeToolsValue>()
    for key in keyedContainer.allKeys {
      object[Key(rawValue: key.stringValue)] = try keyedContainer.decode(
        EdgeToolsValue.self,
        forKey: key
      )
    }
    self = .object(object)
  }
}

// MARK: - Literals

extension EdgeToolsGenerationSchema: ExpressibleByBooleanLiteral {
  public init(booleanLiteral value: Bool) {
    self = .boolean(value)
  }
}

extension EdgeToolsGenerationSchema: ExpressibleByArrayLiteral {
  public init(arrayLiteral elements: Self...) {
    self.init(elements)
  }
}

extension EdgeToolsGenerationSchema: ExpressibleByDictionaryLiteral {
  public init(dictionaryLiteral elements: (Key, EdgeToolsValue)...) {
    self = .object(OrderedDictionary(uniqueKeysWithValues: elements))
  }
}

// MARK: - Raw helpers

extension EdgeToolsGenerationSchema {
  public static func raw(_ key: Key, _ value: EdgeToolsValue) -> Self {
    Self([key: value])
  }

  public var edgeToolsValue: EdgeToolsValue {
    switch self {
    case .boolean(let value):
      .boolean(value)
    case .object(let object):
      .object(OrderedDictionary(uniqueKeysWithValues: object.map { ($0.key.rawValue, $0.value) }))
    }
  }

  static func schemaArrayValue(_ schemas: [Self]) -> EdgeToolsValue {
    .array(schemas.map(\.edgeToolsValue))
  }

  static func schemaObjectValue(
    _ properties: OrderedDictionary<String, Self>
  ) -> EdgeToolsValue {
    .object(
      OrderedDictionary(uniqueKeysWithValues: properties.map { ($0.key, $0.value.edgeToolsValue) })
    )
  }

  static func schemaObjectValue(
    _ properties: KeyValuePairs<String, Self>
  ) -> EdgeToolsValue {
    .object(
      OrderedDictionary(uniqueKeysWithValues: properties.map { ($0.0, $0.1.edgeToolsValue) })
    )
  }

  static func rawStringArray(_ strings: [String]) -> EdgeToolsValue {
    .array(strings.map(EdgeToolsValue.string))
  }

  static func rawStringObject(_ object: OrderedDictionary<String, [String]>)
    -> EdgeToolsValue
  {
    .object(
      OrderedDictionary(
        uniqueKeysWithValues: object.map { key, values in
          (key, Self.rawStringArray(values))
        }
      )
    )
  }

  static func valueType(from value: EdgeToolsValue) -> ValueType? {
    switch value {
    case .string(let string):
      switch string {
      case "string": .string
      case "number": .number
      case "integer": .integer
      case "boolean": .boolean
      case "object": .object
      case "array": .array
      case "null": .null
      default: nil
      }
    case .array(let values):
      values.reduce(into: ValueType()) { partialResult, value in
        guard let type = Self.valueType(from: value) else { return }
        partialResult.formUnion(type)
      }
    default:
      nil
    }
  }
}

// MARK: - Standard keys

extension EdgeToolsGenerationSchema.Key {
  public static let type: Self = "type"
  public static let title: Self = "title"
  public static let description: Self = "description"
  public static let `default`: Self = "default"
  public static let examples: Self = "examples"
  public static let `enum`: Self = "enum"
  public static let const: Self = "const"
  public static let allOf: Self = "allOf"
  public static let anyOf: Self = "anyOf"
  public static let oneOf: Self = "oneOf"
  public static let not: Self = "not"
  public static let `if`: Self = "if"
  public static let then: Self = "then"
  public static let `else`: Self = "else"
  public static let properties: Self = "properties"
  public static let required: Self = "required"
  public static let additionalProperties: Self = "additionalProperties"
  public static let patternProperties: Self = "patternProperties"
  public static let propertyNames: Self = "propertyNames"
  public static let minProperties: Self = "minProperties"
  public static let maxProperties: Self = "maxProperties"
  public static let dependentRequired: Self = "dependentRequired"
  public static let items: Self = "items"
  public static let prefixItems: Self = "prefixItems"
  public static let minItems: Self = "minItems"
  public static let maxItems: Self = "maxItems"
  public static let uniqueItems: Self = "uniqueItems"
  public static let contains: Self = "contains"
  public static let minContains: Self = "minContains"
  public static let maxContains: Self = "maxContains"
  public static let minLength: Self = "minLength"
  public static let maxLength: Self = "maxLength"
  public static let pattern: Self = "pattern"
  public static let multipleOf: Self = "multipleOf"
  public static let minimum: Self = "minimum"
  public static let exclusiveMinimum: Self = "exclusiveMinimum"
  public static let maximum: Self = "maximum"
  public static let exclusiveMaximum: Self = "exclusiveMaximum"
}
