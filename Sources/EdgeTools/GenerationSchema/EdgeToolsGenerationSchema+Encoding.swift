import OrderedCollections
import yyjson

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
