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

