import OrderedCollections
import yyjson

#if Foundation
  import Foundation
#endif

#if canImport(CoreGraphics)
  import CoreGraphics
#endif

// MARK: - EdgeToolsValue

/// A JSON value used for generation schemas.
public enum EdgeToolsValue: Hashable, Sendable {
  /// A string value.
  case string(String)

  /// A boolean value.
  case boolean(Bool)

  /// An array value.
  case array([Self])

  /// An object value.
  case object(OrderedDictionary<String, Self>)

  /// A numerical value.
  case number(Double)

  /// An integer value.
  case integer(Int)

  /// A null value.
  case null

  /// The ``EdgeToolsGenerationSchema/ValueType`` of this value.
  public var type: EdgeToolsGenerationSchema.ValueType {
    switch self {
    case .string: .string
    case .boolean: .boolean
    case .array: .array
    case .object: .object
    case .number: .number
    case .integer: .integer
    case .null: .null
    }
  }
}

// MARK: - Encodable

extension EdgeToolsValue: Encodable {
  public func encode(to encoder: any Encoder) throws {
    switch self {
    case .array(let array):
      var container = encoder.unkeyedContainer()
      for value in array {
        try container.encode(value)
      }
    case .boolean(let value):
      var container = encoder.singleValueContainer()
      try container.encode(value)
    case .integer(let integer):
      var container = encoder.singleValueContainer()
      try container.encode(integer)
    case .null:
      var container = encoder.singleValueContainer()
      try container.encodeNil()
    case .number(let number):
      var container = encoder.singleValueContainer()
      try container.encode(number)
    case .object(let object):
      var container = encoder.container(keyedBy: DynamicCodingKey.self)
      for (key, value) in object {
        try container.encode(value, forKey: DynamicCodingKey(key))
      }
    case .string(let string):
      var container = encoder.singleValueContainer()
      try container.encode(string)
    }
  }
}

// MARK: - Decodable

extension EdgeToolsValue: Decodable {
  public init(from decoder: any Decoder) throws {
    if let keyedContainer = try? decoder.container(keyedBy: DynamicCodingKey.self) {
      var object = OrderedDictionary<String, Self>()
      for key in keyedContainer.allKeys {
        object[key.stringValue] = try keyedContainer.decode(Self.self, forKey: key)
      }
      self = .object(object)
      return
    }

    let unkeyedContainer = try? decoder.unkeyedContainer()
    if var unkeyedContainer {
      var values = [Self]()
      while !unkeyedContainer.isAtEnd {
        values.append(try unkeyedContainer.decode(Self.self))
      }
      self = .array(values)
      return
    }

    let container = try decoder.singleValueContainer()
    if let bool = try? container.decode(Bool.self) {
      self = .boolean(bool)
    } else if let integer = try? container.decode(Int.self) {
      self = .integer(integer)
    } else if let number = try? container.decode(Double.self) {
      self = .number(number)
    } else if let string = try? container.decode(String.self) {
      self = .string(string)
    } else if container.decodeNil() {
      self = .null
    } else {
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid value.")
    }
  }
}

// MARK: - ExpressibleByStringLiteral

extension EdgeToolsValue: ExpressibleByStringLiteral {
  public init(stringLiteral value: String) {
    self = .string(value)
  }
}

// MARK: - ExpressibleByBooleanLiteral

extension EdgeToolsValue: ExpressibleByBooleanLiteral {
  public init(booleanLiteral value: Bool) {
    self = .boolean(value)
  }
}

// MARK: - ExpressibleByFloatLiteral

extension EdgeToolsValue: ExpressibleByFloatLiteral {
  public init(floatLiteral value: Double) {
    self = .number(value)
  }
}

// MARK: - ExpressibleByIntegerLiteral

extension EdgeToolsValue: ExpressibleByIntegerLiteral {
  public init(integerLiteral value: Int) {
    self = .integer(value)
  }
}

// MARK: - ExpressibleByArrayLiteral

extension EdgeToolsValue: ExpressibleByArrayLiteral {
  public init(arrayLiteral elements: Self...) {
    self = .array(elements)
  }
}

// MARK: - ExpressibleByDictionaryLiteral

extension EdgeToolsValue: ExpressibleByDictionaryLiteral {
  public init(dictionaryLiteral elements: (String, Self)...) {
    self = .object(OrderedDictionary(uniqueKeysWithValues: elements))
  }
}

// MARK: - ExpressibleByNilLiteral

extension EdgeToolsValue: ExpressibleByNilLiteral {
  public init(nilLiteral: ()) {
    self = .null
  }
}

struct DynamicCodingKey: CodingKey, Hashable, Sendable {
  let stringValue: String
  let intValue: Int?

  init(_ stringValue: String) {
    self.stringValue = stringValue
    self.intValue = nil
  }

  init?(stringValue: String) {
    self.init(stringValue)
  }

  init?(intValue: Int) {
    self.stringValue = String(intValue)
    self.intValue = intValue
  }
}

// MARK: - EdgeToolsGenerable

/// A type that can generate a generation schema description of itself.
public protocol EdgeToolsGenerable: ConvertibleFromEdgeToolsValue, ConvertibleToEdgeToolsValue {
  /// The generation schema describing this type.
  static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { get }
}

// MARK: - ConvertibleFromEdgeToolsValue

public protocol ConvertibleFromEdgeToolsValue {
  associatedtype EdgeToolsConversionFailure: Error

  /// Creates this type from a ``EdgeToolsValue``.
  init(edgeToolsValue: EdgeToolsValue) throws(EdgeToolsConversionFailure)
}

extension EdgeToolsValue: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) {
    self = edgeToolsValue
  }
}

// MARK: - ConvertibleToEdgeToolsValue

/// A type that can produce an ``EdgeToolsValue`` representation of itself.
public protocol ConvertibleToEdgeToolsValue {
  /// An ``EdgeToolsValue`` representation of this value.
  var edgeToolsValue: EdgeToolsValue { get }
}

extension EdgeToolsValue: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue { self }
}

// MARK: - EdgeToolsValue

extension EdgeToolsValue: EdgeToolsGenerable {
  /// The universal generation schema: a JSON Schema `true`, which accepts any
  /// well-formed ``EdgeToolsValue``.
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .universal }
}

// MARK: - Scalar Types

extension String: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .string(let string) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .string, received: edgeToolsValue.type)
    }
    self = string
  }

  public var edgeToolsValue: EdgeToolsValue { .string(self) }
}

extension Bool: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .boolean }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .boolean(let boolean) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .boolean, received: edgeToolsValue.type)
    }
    self = boolean
  }

  public var edgeToolsValue: EdgeToolsValue { .boolean(self) }
}

extension Double: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    switch edgeToolsValue {
    case .number(let number):
      self = number
    case .integer(let integer):
      self = Double(integer)
    default:
      throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
    }
  }

  public var edgeToolsValue: EdgeToolsValue { .number(self) }
}

extension Float: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    switch edgeToolsValue {
    case .number(let number):
      self = Float(number)
    case .integer(let integer):
      self = Float(integer)
    default:
      throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
    }
  }

  public var edgeToolsValue: EdgeToolsValue { .number(Double(self)) }
}

extension EdgeToolsGenerable where Self: FixedWidthInteger, Self: SignedInteger {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }

  public var edgeToolsValue: EdgeToolsValue { .integer(Int(self)) }
}

extension EdgeToolsGenerable where Self: FixedWidthInteger, Self: UnsignedInteger {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }

  public var edgeToolsValue: EdgeToolsValue {
    let asInt = Int(truncatingIfNeeded: self)
    precondition(
      asInt >= 0,
      "\(Self.self) value \(self) exceeds Int.max and cannot be encoded as an EdgeToolsValue integer."
    )
    return .integer(asInt)
  }
}

extension Int8: EdgeToolsGenerable {}
extension Int16: EdgeToolsGenerable {}
extension Int32: EdgeToolsGenerable {}
extension Int64: EdgeToolsGenerable {}
extension Int: EdgeToolsGenerable {}
extension UInt8: EdgeToolsGenerable {}
extension UInt16: EdgeToolsGenerable {}
extension UInt32: EdgeToolsGenerable {}
extension UInt64: EdgeToolsGenerable {}
extension UInt: EdgeToolsGenerable {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension Int128: EdgeToolsGenerable {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension UInt128: EdgeToolsGenerable {}

// MARK: - Foundation

#if Foundation
  extension Data: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      guard case .string(let string) = edgeToolsValue else {
        throw EdgeToolsValueTypeError(expected: .string, received: edgeToolsValue.type)
      }
      self = Data(string.utf8)
    }

    public var edgeToolsValue: EdgeToolsValue {
      .string(String(decoding: self, as: UTF8.self))
    }
  }

  extension Decimal: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      switch edgeToolsValue {
      case .number(let number):
        self = Decimal(number)
      case .integer(let integer):
        self = Decimal(integer)
      default:
        throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
      }
    }

    public var edgeToolsValue: EdgeToolsValue {
      .string("\(self)")
    }
  }
#endif

// MARK: - Generic Containers

extension Array: EdgeToolsGenerable where Element: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.type(.array), .items(Element.edgeToolsGenerationSchema))
  }
}

extension Array: ConvertibleFromEdgeToolsValue where Element: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .array(let array) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .array, received: edgeToolsValue.type)
    }
    self = try array.map { try Element(edgeToolsValue: $0) }
  }
}

extension Array: ConvertibleToEdgeToolsValue where Element: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue { .array(self.map(\.edgeToolsValue)) }
}

extension Dictionary: EdgeToolsGenerable
where Key == String, Value: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(
      .type(.object),
      .additionalProperties(Value.edgeToolsGenerationSchema)
    )
  }
}

extension Dictionary: ConvertibleFromEdgeToolsValue
where Key == String, Value: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .object(let object) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .object, received: edgeToolsValue.type)
    }
    self = try Dictionary(
      uniqueKeysWithValues: object.map { key, value in
        (key, try Value(edgeToolsValue: value))
      }
    )
  }
}

extension Dictionary: ConvertibleToEdgeToolsValue
where Key == String, Value: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue {
    .object(
      OrderedDictionary(
        uniqueKeysWithValues: self.map { key, value in (key, value.edgeToolsValue) }
      )
    )
  }
}

extension Optional: EdgeToolsGenerable where Wrapped: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    Wrapped.edgeToolsGenerationSchema.nullable()
  }
}

extension Optional: ConvertibleFromEdgeToolsValue where Wrapped: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    switch edgeToolsValue {
    case .null:
      self = nil
    default:
      self = try Wrapped(edgeToolsValue: edgeToolsValue)
    }
  }
}

extension Optional: ConvertibleToEdgeToolsValue where Wrapped: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue {
    switch self {
    case .none: .null
    case .some(let wrapped): wrapped.edgeToolsValue
    }
  }
}

#if canImport(CoreGraphics)
  extension CGFloat: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      switch edgeToolsValue {
      case .number(let number):
        self = CGFloat(number)
      case .integer(let integer):
        self = CGFloat(integer)
      default:
        throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
      }
    }

    public var edgeToolsValue: EdgeToolsValue { .number(Double(self)) }
  }
#endif

// MARK: - EdgeToolsValueTypeError

public struct EdgeToolsValueTypeError: Error, Hashable, Sendable {
  public let expected: EdgeToolsGenerationSchema.ValueType
  public let received: EdgeToolsGenerationSchema.ValueType

  public init(
    expected: EdgeToolsGenerationSchema.ValueType,
    received: EdgeToolsGenerationSchema.ValueType
  ) {
    self.expected = expected
    self.received = received
  }
}

// MARK: - Helpers

extension FixedWidthInteger {
  fileprivate static func integer(from edgeToolsValue: EdgeToolsValue) throws -> Self {
    guard case .integer(let integer) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .integer, received: edgeToolsValue.type)
    }
    guard let value = Self(exactly: integer) else {
      throw EdgeToolsValueTypeError(expected: .integer, received: edgeToolsValue.type)
    }
    return value
  }
}

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

  private static func rawStringObject(_ object: OrderedDictionary<String, [String]>)
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

// MARK: - Fragment helpers

extension EdgeToolsGenerationSchema {
  public static var string: Self { .type(.string) }
  public static var number: Self { .type(.number) }
  public static var integer: Self { .type(.integer) }
  public static var null: Self { .type(.null) }
  public static var boolean: Self { .type(.boolean) }

  /// The universal generation schema, equivalent to JSON Schema `true` — it
  /// accepts any well-formed value.
  public static var universal: Self { .boolean(true) }

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

// MARK: - Ordered key encoding

extension EdgeToolsGenerationSchema {
  public func orderedKeyEncoded() -> String {
    OrderedKeyJSONWriter.encode(self.edgeToolsValue)
  }
}

package struct OrderedKeyJSONWriter: ~Copyable {
  let document: UnsafeMutablePointer<yyjson_mut_doc>

  deinit { yyjson_mut_doc_free(self.document) }

  package static func encode(_ value: EdgeToolsValue) -> String {
    let document = yyjson_mut_doc_new(nil)!
    let writer = Self(document: document)
    let root = writer.value(value)
    yyjson_mut_doc_set_root(writer.document, root)

    var length = 0
    let output = yyjson_mut_write(writer.document, YYJSON_WRITE_INF_AND_NAN_AS_NULL, &length)!
    defer { output.deallocate() }

    let bytes = UnsafeRawPointer(output).assumingMemoryBound(to: UInt8.self)
    let buffer = UnsafeBufferPointer<UInt8>(start: bytes, count: length)
    return String(bytes: buffer, encoding: .utf8)!
  }

  func value(_ value: EdgeToolsValue) -> UnsafeMutablePointer<yyjson_mut_val>? {
    switch value {
    case .null: yyjson_mut_null(self.document)
    case .boolean(let value): yyjson_mut_bool(self.document, value)
    case .integer(let value): yyjson_mut_sint(self.document, Int64(value))
    case .number(let value): yyjson_mut_real(self.document, value)
    case .string(let value): self.string(value)
    case .array(let values): self.array(values)
    case .object(let object): self.object(object)
    }
  }

  func string(_ value: String) -> UnsafeMutablePointer<yyjson_mut_val>? {
    value.withCString { characters in
      yyjson_mut_strncpy(self.document, characters, value.utf8.count)
    }
  }

  private func array(_ values: [EdgeToolsValue]) -> UnsafeMutablePointer<yyjson_mut_val>? {
    guard let array = yyjson_mut_arr(self.document) else { return nil }
    for value in values {
      guard let encodedValue = self.value(value), yyjson_mut_arr_add_val(array, encodedValue)
      else { return nil }
    }
    return array
  }

  private func object(
    _ object: OrderedDictionary<String, EdgeToolsValue>
  ) -> UnsafeMutablePointer<yyjson_mut_val>? {
    guard let encodedObject = yyjson_mut_obj(self.document) else { return nil }
    for (key, value) in object {
      guard let encodedKey = self.string(key),
        let encodedValue = self.value(value),
        yyjson_mut_obj_add(encodedObject, encodedKey, encodedValue)
      else { return nil }
    }
    return encodedObject
  }
}

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

  // pi-lens-ignore: file_length
}
