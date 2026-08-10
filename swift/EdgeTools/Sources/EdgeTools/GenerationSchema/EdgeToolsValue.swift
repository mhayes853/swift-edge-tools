import OrderedCollections
import yyjson

#if Foundation
  import _EdgeToolsFoundation
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

// MARK: - Codable

extension EdgeToolsValue: EdgeToolsEncodable {
  #if !$Embedded
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
  #endif
}

extension EdgeToolsValue: EdgeToolsDecodable {
  #if !$Embedded
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
  #endif
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

// MARK: - JSON Decoding

extension EdgeToolsValue {
  public init(json bytes: [UInt8]) throws {
    guard !bytes.isEmpty else { throw EdgeToolsError.emptyJSONInput }

    var bytes = bytes
    var error = yyjson_read_err()
    let document = bytes.withUnsafeMutableBytes { buffer in
      yyjson_read_opts(
        buffer.baseAddress?.assumingMemoryBound(to: CChar.self),
        buffer.count,
        YYJSON_READ_NOFLAG,
        nil,
        &error
      )
    }
    guard let document, let root = yyjson_doc_get_root(document) else {
      let message = error.msg.map(String.init(cString:)) ?? "Invalid JSON."
      throw EdgeToolsError(
        code: .invalidJSON,
        message: "\(message) (byte offset \(error.pos))."
      )
    }
    defer { yyjson_doc_free(document) }
    self = try readEdgeToolsValue(from: root)
  }

  package init(json string: String) throws {
    try self.init(json: Array(string.utf8))
  }
}

// MARK: - JSON Encoding

extension ConvertibleToEdgeToolsValue {
  public func orderedJSONString() -> String {
    OrderedKeyJSONWriter.encode(self.edgeToolsValue)
  }
}

// MARK: - JSON Reading

private func readEdgeToolsValue(
  from value: UnsafeMutablePointer<yyjson_val>
) throws -> EdgeToolsValue {
  if yyjson_is_null(value) {
    return .null
  } else if yyjson_is_bool(value) {
    return .boolean(yyjson_get_bool(value))
  } else if yyjson_is_str(value) {
    return .string(string(from: value))
  } else if yyjson_is_sint(value) {
    guard let integer = Int(exactly: yyjson_get_sint(value)) else {
      throw EdgeToolsError.jsonIntegerOutOfRange
    }
    return .integer(integer)
  } else if yyjson_is_uint(value) {
    guard let integer = Int(exactly: yyjson_get_uint(value)) else {
      throw EdgeToolsError.jsonIntegerOutOfRange
    }
    return .integer(integer)
  } else if yyjson_is_real(value) {
    let number = yyjson_get_real(value)
    guard number.isFinite else { throw EdgeToolsError.nonFiniteJSONNumber }
    return .number(number)
  } else if yyjson_is_arr(value) {
    return .array(try array(from: value))
  } else if yyjson_is_obj(value) {
    return .object(try object(from: value))
  }
  throw EdgeToolsError.invalidJSONValue
}

private func array(
  from value: UnsafeMutablePointer<yyjson_val>
) throws -> [EdgeToolsValue] {
  var values = [EdgeToolsValue]()
  values.reserveCapacity(Int(yyjson_arr_size(value)))
  var iterator = yyjson_arr_iter()
  yyjson_arr_iter_init(value, &iterator)
  while let element = yyjson_arr_iter_next(&iterator) {
    values.append(try readEdgeToolsValue(from: element))
  }
  return values
}

private func object(
  from value: UnsafeMutablePointer<yyjson_val>
) throws -> OrderedDictionary<String, EdgeToolsValue> {
  var object = OrderedDictionary<String, EdgeToolsValue>()
  object.reserveCapacity(Int(yyjson_obj_size(value)))
  var iterator = yyjson_obj_iter()
  yyjson_obj_iter_init(value, &iterator)
  while let key = yyjson_obj_iter_next(&iterator) {
    guard let element = yyjson_obj_iter_get_val(key) else { continue }
    object[string(from: key)] = try readEdgeToolsValue(from: element)
  }
  return object
}

private func string(from value: UnsafeMutablePointer<yyjson_val>) -> String {
  guard let characters = yyjson_get_str(value) else { return "" }
  let length = Int(yyjson_get_len(value))
  let buffer = UnsafeRawPointer(characters).assumingMemoryBound(to: UInt8.self)
  return String(decoding: UnsafeBufferPointer(start: buffer, count: length), as: UTF8.self)
}
