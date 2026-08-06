#if !$Embedded
  import yyjson

  package struct EdgeToolsJSONDecoder {
    package var userInfo: [CodingUserInfoKey: Any]

    package init(userInfo: [CodingUserInfoKey: Any] = [:]) {
      self.userInfo = userInfo
    }

    package func decode<Value: Decodable>(
      _ type: Value.Type,
      from bytes: [UInt8]
    ) throws -> Value {
      var bytes = bytes
      var error = yyjson_read_err()
      let pointer = bytes.withUnsafeMutableBytes { buffer in
        yyjson_read_opts(
          buffer.baseAddress?.assumingMemoryBound(to: CChar.self),
          buffer.count,
          YYJSON_READ_NOFLAG,
          nil,
          &error
        )
      }
      guard let pointer, let root = yyjson_doc_get_root(pointer) else {
        let message = error.msg.map(String.init(cString:)) ?? "Invalid JSON."
        throw DecodingError.dataCorrupted(
          DecodingError.Context(
            codingPath: [],
            debugDescription: "\(message) (byte offset \(error.pos))."
          )
        )
      }

      let document = YYJSONDocument(pointer: pointer)
      let decoder = YYJSONValueDecoder(
        document: document,
        value: root,
        codingPath: [],
        userInfo: self.userInfo
      )
      return try Value(from: decoder)
    }
  }

  private final class YYJSONDocument {
    let pointer: UnsafeMutablePointer<yyjson_doc>

    init(pointer: UnsafeMutablePointer<yyjson_doc>) {
      self.pointer = pointer
    }

    deinit {
      yyjson_doc_free(self.pointer)
    }
  }

  private struct YYJSONValueDecoder: Decoder {
    let document: YYJSONDocument
    let value: UnsafeMutablePointer<yyjson_val>
    let codingPath: [any CodingKey]
    let userInfo: [CodingUserInfoKey: Any]

    func container<Key: CodingKey>(
      keyedBy type: Key.Type
    ) throws -> KeyedDecodingContainer<Key> {
      guard yyjson_is_obj(self.value) else { throw self.typeMismatch([String: Any].self) }
      return KeyedDecodingContainer(YYJSONKeyedDecodingContainer(decoder: self))
    }

    func unkeyedContainer() throws -> any UnkeyedDecodingContainer {
      guard yyjson_is_arr(self.value) else { throw self.typeMismatch([Any].self) }
      return YYJSONUnkeyedDecodingContainer(decoder: self)
    }

    func singleValueContainer() throws -> any SingleValueDecodingContainer {
      YYJSONSingleValueDecodingContainer(decoder: self)
    }

    func nested(
      value: UnsafeMutablePointer<yyjson_val>,
      at key: any CodingKey
    ) -> Self {
      Self(
        document: self.document,
        value: value,
        codingPath: self.codingPath + [key],
        userInfo: self.userInfo
      )
    }

    func typeMismatch(_ type: Any.Type) -> DecodingError {
      DecodingError.typeMismatch(
        type,
        DecodingError.Context(
          codingPath: self.codingPath,
          debugDescription: "Expected \(type), but found \(self.valueDescription)."
        )
      )
    }

    static func string(from value: UnsafeMutablePointer<yyjson_val>) -> String {
      let count = Int(yyjson_get_len(value))
      guard let characters = yyjson_get_str(value) else { return "" }
      let bytes = UnsafeRawPointer(characters).assumingMemoryBound(to: UInt8.self)
      return String(decoding: UnsafeBufferPointer(start: bytes, count: count), as: UTF8.self)
    }

    private var valueDescription: String {
      if yyjson_is_null(self.value) { "null" }
      else if yyjson_is_bool(self.value) { "a boolean" }
      else if yyjson_is_num(self.value) { "a number" }
      else if yyjson_is_str(self.value) { "a string" }
      else if yyjson_is_arr(self.value) { "an array" }
      else if yyjson_is_obj(self.value) { "an object" }
      else { "an invalid JSON value" }
    }
  }

  private struct YYJSONKeyedDecodingContainer<Key: CodingKey>: KeyedDecodingContainerProtocol {
    let decoder: YYJSONValueDecoder
    let values: [String: UnsafeMutablePointer<yyjson_val>]
    let allKeys: [Key]

    var codingPath: [any CodingKey] { self.decoder.codingPath }

    init(decoder: YYJSONValueDecoder) {
      self.decoder = decoder
      var values = [String: UnsafeMutablePointer<yyjson_val>]()
      var keys = [Key]()
      var iterator = yyjson_obj_iter()
      yyjson_obj_iter_init(decoder.value, &iterator)
      while let key = yyjson_obj_iter_next(&iterator), let value = yyjson_obj_iter_get_val(key) {
        let string = YYJSONValueDecoder.string(from: key)
        values[string] = value
        if let key = Key(stringValue: string) { keys.append(key) }
      }
      self.values = values
      self.allKeys = keys
    }

    func contains(_ key: Key) -> Bool {
      self.values[key.stringValue] != nil
    }

    func decodeNil(forKey key: Key) throws -> Bool {
      yyjson_is_null(try self.value(forKey: key))
    }

    func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
      try T(from: self.nestedDecoder(forKey: key))
    }

    func nestedContainer<NestedKey: CodingKey>(
      keyedBy type: NestedKey.Type,
      forKey key: Key
    ) throws -> KeyedDecodingContainer<NestedKey> {
      try self.nestedDecoder(forKey: key).container(keyedBy: type)
    }

    func nestedUnkeyedContainer(forKey key: Key) throws -> any UnkeyedDecodingContainer {
      try self.nestedDecoder(forKey: key).unkeyedContainer()
    }

    func superDecoder() throws -> any Decoder {
      guard let key = Key(stringValue: "super") else { return self.decoder }
      return try self.superDecoder(forKey: key)
    }

    func superDecoder(forKey key: Key) throws -> any Decoder {
      try self.nestedDecoder(forKey: key)
    }

    private func value(forKey key: Key) throws -> UnsafeMutablePointer<yyjson_val> {
      guard let value = self.values[key.stringValue] else {
        throw DecodingError.keyNotFound(
          key,
          DecodingError.Context(
            codingPath: self.codingPath,
            debugDescription: "No value associated with key \"\(key.stringValue)\"."
          )
        )
      }
      return value
    }

    private func nestedDecoder(forKey key: Key) throws -> YYJSONValueDecoder {
      self.decoder.nested(value: try self.value(forKey: key), at: key)
    }
  }

  private struct YYJSONUnkeyedDecodingContainer: UnkeyedDecodingContainer {
    let decoder: YYJSONValueDecoder
    let count: Int?
    var currentIndex = 0

    var codingPath: [any CodingKey] { self.decoder.codingPath }
    var isAtEnd: Bool { self.currentIndex >= self.count ?? 0 }

    init(decoder: YYJSONValueDecoder) {
      self.decoder = decoder
      self.count = Int(yyjson_arr_size(decoder.value))
    }

    mutating func decodeNil() throws -> Bool {
      guard !self.isAtEnd else { throw self.valueNotFound() }
      guard yyjson_is_null(self.currentValue) else { return false }
      self.currentIndex += 1
      return true
    }

    mutating func decode<T: Decodable>(_ type: T.Type) throws -> T {
      let decoder = try self.nextDecoder()
      return try T(from: decoder)
    }

    mutating func nestedContainer<NestedKey: CodingKey>(
      keyedBy type: NestedKey.Type
    ) throws -> KeyedDecodingContainer<NestedKey> {
      try self.nextDecoder().container(keyedBy: type)
    }

    mutating func nestedUnkeyedContainer() throws -> any UnkeyedDecodingContainer {
      try self.nextDecoder().unkeyedContainer()
    }

    mutating func superDecoder() throws -> any Decoder {
      try self.nextDecoder()
    }

    private var currentValue: UnsafeMutablePointer<yyjson_val>? {
      yyjson_arr_get(self.decoder.value, self.currentIndex)
    }

    private mutating func nextDecoder() throws -> YYJSONValueDecoder {
      guard !self.isAtEnd, let value = self.currentValue else { throw self.valueNotFound() }
      let key = YYJSONIndexKey(index: self.currentIndex)
      self.currentIndex += 1
      return self.decoder.nested(value: value, at: key)
    }

    private func valueNotFound() -> DecodingError {
      DecodingError.valueNotFound(
        Any.self,
        DecodingError.Context(
          codingPath: self.codingPath + [YYJSONIndexKey(index: self.currentIndex)],
          debugDescription: "Unkeyed container is at end."
        )
      )
    }
  }

  private struct YYJSONSingleValueDecodingContainer: SingleValueDecodingContainer {
    let decoder: YYJSONValueDecoder

    var codingPath: [any CodingKey] { self.decoder.codingPath }

    func decodeNil() -> Bool {
      yyjson_is_null(self.decoder.value)
    }

    func decode(_ type: Bool.Type) throws -> Bool {
      guard yyjson_is_bool(self.decoder.value) else { throw self.decoder.typeMismatch(type) }
      return yyjson_get_bool(self.decoder.value)
    }

    func decode(_ type: String.Type) throws -> String {
      guard yyjson_is_str(self.decoder.value) else { throw self.decoder.typeMismatch(type) }
      return YYJSONValueDecoder.string(from: self.decoder.value)
    }

    func decode(_ type: Double.Type) throws -> Double { try self.floatingPoint(type) }
    func decode(_ type: Float.Type) throws -> Float { try self.floatingPoint(type) }
    func decode(_ type: Int.Type) throws -> Int { try self.integer(type) }
    func decode(_ type: Int8.Type) throws -> Int8 { try self.integer(type) }
    func decode(_ type: Int16.Type) throws -> Int16 { try self.integer(type) }
    func decode(_ type: Int32.Type) throws -> Int32 { try self.integer(type) }
    func decode(_ type: Int64.Type) throws -> Int64 { try self.integer(type) }
    func decode(_ type: UInt.Type) throws -> UInt { try self.integer(type) }
    func decode(_ type: UInt8.Type) throws -> UInt8 { try self.integer(type) }
    func decode(_ type: UInt16.Type) throws -> UInt16 { try self.integer(type) }
    func decode(_ type: UInt32.Type) throws -> UInt32 { try self.integer(type) }
    func decode(_ type: UInt64.Type) throws -> UInt64 { try self.integer(type) }

    func decode<T: Decodable>(_ type: T.Type) throws -> T {
      try T(from: self.decoder)
    }

    private func integer<Integer: FixedWidthInteger>(_ type: Integer.Type) throws -> Integer {
      let value: Integer?
      if yyjson_is_sint(self.decoder.value) {
        value = Integer(exactly: yyjson_get_sint(self.decoder.value))
      } else if yyjson_is_uint(self.decoder.value) {
        value = Integer(exactly: yyjson_get_uint(self.decoder.value))
      } else {
        throw self.decoder.typeMismatch(type)
      }
      guard let value else { throw self.decoder.typeMismatch(type) }
      return value
    }

    private func floatingPoint<Value: BinaryFloatingPoint>(_ type: Value.Type) throws -> Value {
      guard yyjson_is_num(self.decoder.value) else { throw self.decoder.typeMismatch(type) }
      let value = Value(yyjson_get_num(self.decoder.value))
      guard value.isFinite else { throw self.decoder.typeMismatch(type) }
      return value
    }
  }

  private struct YYJSONIndexKey: CodingKey {
    let intValue: Int?
    let stringValue: String

    init(index: Int) {
      self.intValue = index
      self.stringValue = "Index \(index)"
    }

    init?(stringValue: String) { return nil }
    init?(intValue: Int) { self.init(index: intValue) }
  }
#endif
