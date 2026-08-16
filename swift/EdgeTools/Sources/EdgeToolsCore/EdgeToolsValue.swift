import OrderedCollections
import yyjson

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
}

// MARK: - Case Values

extension EdgeToolsValue {
  public var string: String? {
    guard case .string(let value) = self else {
      return nil
    }
    return value
  }

  public var boolean: Bool? {
    guard case .boolean(let value) = self else {
      return nil
    }
    return value
  }

  public var array: [Self]? {
    guard case .array(let value) = self else {
      return nil
    }
    return value
  }

  public var object: OrderedDictionary<String, Self>? {
    guard case .object(let value) = self else {
      return nil
    }
    return value
  }

  public var number: Double? {
    guard case .number(let value) = self else {
      return nil
    }
    return value
  }

  public var integer: Int? {
    guard case .integer(let value) = self else {
      return nil
    }
    return value
  }

  public var double: Double? {
    switch self {
    case .integer(let value): Double(value)
    case .number(let value): value
    default: nil
    }
  }

  public var isNull: Bool {
    if case .null = self {
      return true
    }
    return false
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

extension EdgeToolsValue {
  public func orderedJSONString() -> String {
    OrderedKeyJSONWriter.encode(self)
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

// MARK: - Ordered Key JSON Writer

private struct OrderedKeyJSONWriter: ~Copyable {
  let document: UnsafeMutablePointer<yyjson_mut_doc>

  deinit { yyjson_mut_doc_free(self.document) }

  static func encode(_ value: EdgeToolsValue) -> String {
    let document = yyjson_mut_doc_new(nil)!
    let writer = Self(document: document)
    let root = writer.value(value)
    yyjson_mut_doc_set_root(writer.document, root)

    var length = 0
    let output = yyjson_mut_write(writer.document, YYJSON_WRITE_INF_AND_NAN_AS_NULL, &length)!
    defer { output.deallocate() }

    let bytes = UnsafeRawPointer(output).assumingMemoryBound(to: UInt8.self)
    let buffer = UnsafeBufferPointer<UInt8>(start: bytes, count: length)
    return String(decoding: buffer, as: UTF8.self)
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
