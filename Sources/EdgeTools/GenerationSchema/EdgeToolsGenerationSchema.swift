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
    self = schemas.reduce(Self.object(OrderedDictionary<Key, EdgeToolsValue>())) {
      partialResult,
      schema in
      partialResult.merging(schema)
    }
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

  private static func schemaArrayValue(_ schemas: [Self]) -> EdgeToolsValue {
    .array(schemas.map(\.edgeToolsValue))
  }

  private static func schemaObjectValue(
    _ properties: OrderedDictionary<String, Self>
  ) -> EdgeToolsValue {
    .object(
      OrderedDictionary(uniqueKeysWithValues: properties.map { ($0.key, $0.value.edgeToolsValue) })
    )
  }

  private static func schemaObjectValue(
    _ properties: KeyValuePairs<String, Self>
  ) -> EdgeToolsValue {
    .object(
      OrderedDictionary(uniqueKeysWithValues: properties.map { ($0.0, $0.1.edgeToolsValue) })
    )
  }

  private static func rawStringArray(_ strings: [String]) -> EdgeToolsValue {
    .array(strings.map(EdgeToolsValue.string))
  }

  private static func rawStringObject(_ object: OrderedDictionary<String, [String]>) -> EdgeToolsValue
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

// MARK: - Fragment helpers

extension EdgeToolsGenerationSchema {
  public static var string: Self { .type(.string) }
  public static var number: Self { .type(.number) }
  public static var integer: Self { .type(.integer) }
  public static var null: Self { .type(.null) }
  public static var boolean: Self { .type(.boolean) }

  public static func type(_ type: ValueType) -> Self {
    Self([.type: type.edgeToolsValue])
  }

  public static func title(_ value: String) -> Self {
    Self([.title: .string(value)])
  }

  public static func description(_ value: String) -> Self {
    Self([.description: .string(value)])
  }

  public static func `default`(_ value: EdgeToolsValue) -> Self {
    Self([.default: value])
  }

  public static func examples(_ value: [EdgeToolsValue]) -> Self {
    Self([.examples: .array(value)])
  }

  public static func `enum`(_ value: [EdgeToolsValue]) -> Self {
    Self([.enum: .array(value)])
  }

  public static func const(_ value: EdgeToolsValue) -> Self {
    Self([.const: value])
  }

  public static func allOf(_ schemas: [Self]) -> Self {
    Self([.allOf: Self.schemaArrayValue(schemas)])
  }

  public static func anyOf(_ schemas: [Self]) -> Self {
    Self([.anyOf: Self.schemaArrayValue(schemas)])
  }

  public static func oneOf(_ schemas: [Self]) -> Self {
    Self([.oneOf: Self.schemaArrayValue(schemas)])
  }

  public static func not(_ schema: Self) -> Self {
    Self([.not: schema.edgeToolsValue])
  }

  public static func `if`(_ schema: Self) -> Self {
    Self([.if: schema.edgeToolsValue])
  }

  public static func then(_ schema: Self) -> Self {
    Self([.then: schema.edgeToolsValue])
  }

  public static func `else`(_ schema: Self) -> Self {
    Self([.else: schema.edgeToolsValue])
  }

  public static func properties(_ properties: KeyValuePairs<String, Self>) -> Self {
    Self([.properties: Self.schemaObjectValue(properties)])
  }

  public static func properties(_ properties: KeyValuePairs<String, [Self]>) -> Self {
    Self(
      [
        .properties: .object(
          OrderedDictionary(
            uniqueKeysWithValues: properties.map { entry in
              (entry.0, EdgeToolsGenerationSchema(entry.1).edgeToolsValue)
            }
          )
        )
      ]
    )
  }

  public static func property(_ key: String, _ schema: Self) -> Self {
    Self.properties(KeyValuePairs(dictionaryLiteral: (key, schema)))
  }

  public static func property(_ key: String, _ schemas: Self...) -> Self {
    Self.properties(KeyValuePairs(dictionaryLiteral: (key, Self(schemas))))
  }

  public static func required(_ keys: [String]) -> Self {
    Self([.required: Self.rawStringArray(keys)])
  }

  public static func additionalProperties(_ schema: Self) -> Self {
    Self([.additionalProperties: schema.edgeToolsValue])
  }

  public static func additionalProperties(_ allowed: Bool) -> Self {
    Self([.additionalProperties: .boolean(allowed)])
  }

  public static func patternProperties(_ properties: KeyValuePairs<String, Self>) -> Self {
    Self([.patternProperties: Self.schemaObjectValue(properties)])
  }

  public static func propertyNames(_ schema: Self) -> Self {
    Self([.propertyNames: schema.edgeToolsValue])
  }

  public static func minProperties(_ value: Int) -> Self {
    Self([.minProperties: .integer(value)])
  }

  public static func maxProperties(_ value: Int) -> Self {
    Self([.maxProperties: .integer(value)])
  }

  public static func dependentRequired(_ value: KeyValuePairs<String, [String]>) -> Self {
    Self(
      [
        .dependentRequired: Self.rawStringObject(
          OrderedDictionary(uniqueKeysWithValues: value.map { ($0.0, $0.1) })
        )
      ]
    )
  }

  public static func items(_ schema: Self) -> Self {
    Self([.items: schema.edgeToolsValue])
  }

  public static func prefixItems(_ schemas: [Self]) -> Self {
    Self([.prefixItems: Self.schemaArrayValue(schemas)])
  }

  public static func minItems(_ value: Int) -> Self {
    Self([.minItems: .integer(value)])
  }

  public static func maxItems(_ value: Int) -> Self {
    Self([.maxItems: .integer(value)])
  }

  public static func uniqueItems(_ value: Bool = true) -> Self {
    Self([.uniqueItems: .boolean(value)])
  }

  public static func contains(_ schema: Self) -> Self {
    Self([.contains: schema.edgeToolsValue])
  }

  public static func minContains(_ value: Int) -> Self {
    Self([.minContains: .integer(value)])
  }

  public static func maxContains(_ value: Int) -> Self {
    Self([.maxContains: .integer(value)])
  }

  public static func minLength(_ value: Int) -> Self {
    Self([.minLength: .integer(value)])
  }

  public static func maxLength(_ value: Int) -> Self {
    Self([.maxLength: .integer(value)])
  }

  public static func pattern(_ value: String) -> Self {
    Self([.pattern: .string(value)])
  }

  public static func lengthRange(_ range: ClosedRange<Int>) -> Self {
    Self(.minLength(range.lowerBound), .maxLength(range.upperBound))
  }

  public static func lengthRange(_ range: PartialRangeFrom<Int>) -> Self {
    Self(.minLength(range.lowerBound))
  }

  public static func lengthRange(_ range: PartialRangeThrough<Int>) -> Self {
    Self(.maxLength(range.upperBound))
  }

  public static func lengthRange(_ range: Range<Int>) -> Self {
    Self(.minLength(range.lowerBound), .maxLength(range.upperBound - 1))
  }

  public static func multipleOf(_ value: Int) -> Self {
    Self([.multipleOf: .integer(value)])
  }

  public static func multipleOf(_ value: Double) -> Self {
    Self([.multipleOf: .number(value)])
  }

  public static func minimum(_ value: Int) -> Self {
    Self([.minimum: .integer(value)])
  }

  public static func minimum(_ value: Double) -> Self {
    Self([.minimum: .number(value)])
  }

  public static func exclusiveMinimum(_ value: Int) -> Self {
    Self([.exclusiveMinimum: .integer(value)])
  }

  public static func exclusiveMinimum(_ value: Double) -> Self {
    Self([.exclusiveMinimum: .number(value)])
  }

  public static func maximum(_ value: Int) -> Self {
    Self([.maximum: .integer(value)])
  }

  public static func maximum(_ value: Double) -> Self {
    Self([.maximum: .number(value)])
  }

  public static func exclusiveMaximum(_ value: Int) -> Self {
    Self([.exclusiveMaximum: .integer(value)])
  }

  public static func exclusiveMaximum(_ value: Double) -> Self {
    Self([.exclusiveMaximum: .number(value)])
  }

  public static func range(_ range: ClosedRange<Int>) -> Self {
    Self(.minimum(range.lowerBound), .maximum(range.upperBound))
  }

  public static func range(_ range: ClosedRange<Double>) -> Self {
    Self(.minimum(range.lowerBound), .maximum(range.upperBound))
  }

  public static func range(_ range: PartialRangeFrom<Int>) -> Self {
    Self(.minimum(range.lowerBound))
  }

  public static func range(_ range: PartialRangeFrom<Double>) -> Self {
    Self(.minimum(range.lowerBound))
  }

  public static func range(_ range: PartialRangeThrough<Int>) -> Self {
    Self(.maximum(range.upperBound))
  }

  public static func range(_ range: PartialRangeThrough<Double>) -> Self {
    Self(.maximum(range.upperBound))
  }

  public static func range(_ range: Range<Int>) -> Self {
    Self(.minimum(range.lowerBound), .exclusiveMaximum(range.upperBound))
  }

  public static func range(_ range: Range<Double>) -> Self {
    Self(.minimum(range.lowerBound), .exclusiveMaximum(range.upperBound))
  }
}
