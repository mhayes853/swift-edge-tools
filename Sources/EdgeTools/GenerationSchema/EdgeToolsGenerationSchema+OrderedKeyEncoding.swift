import OrderedCollections

// MARK: - Ordered key encoding

extension EdgeToolsGenerationSchema {
  public func orderedKeyEncoded() -> String {
    String(decoding: encodeEdgeToolsJSON(self), as: UTF8.self)
  }
}

package func encodeEdgeToolsJSON(_ schema: EdgeToolsGenerationSchema) -> [UInt8] {
  var writer = EdgeToolsJSONWriter()
  writer.writeSchema(schema)
  return writer.buffer
}

package func encodeEdgeToolsJSON(_ value: EdgeToolsValue) -> [UInt8] {
  var writer = EdgeToolsJSONWriter()
  writer.writeValue(value)
  return writer.buffer
}

package func encodeEdgeToolsJSONString(_ value: String) -> [UInt8] {
  var writer = EdgeToolsJSONWriter()
  writer.writeString(value)
  return writer.buffer
}

private struct EdgeToolsJSONWriter {
  static let openBracket = UInt8(0x5B)
  static let closeBracket = UInt8(0x5D)
  static let openBrace = UInt8(0x7B)
  static let closeBrace = UInt8(0x7D)
  static let comma = UInt8(0x2C)
  static let colon = UInt8(0x3A)
  static let quote = UInt8(0x22)
  static let backslash = UInt8(0x5C)
  static let backspace = UInt8(0x08)
  static let formFeed = UInt8(0x0C)
  static let newline = UInt8(0x0A)
  static let carriageReturn = UInt8(0x0D)
  static let tab = UInt8(0x09)
  static let controlCharLimit = UInt8(0x20)

  static let hexDigitTable = [
    UInt8(0x30), 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66
  ]
  static let zero = UInt8(0x30)

  var buffer = [UInt8]()

  mutating func writeSchema(_ schema: EdgeToolsGenerationSchema) {
    switch schema {
    case .boolean(let value):
      self.writeBool(value)
    case .object(let object):
      self.writeSchemaObject(object)
    }
  }

  private mutating func writeSchemaObject(
    _ object: OrderedDictionary<EdgeToolsGenerationSchema.Key, EdgeToolsValue>
  ) {
    self.buffer.append(Self.openBrace)
    for (offset, entry) in object.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeString(entry.key.rawValue)
      self.buffer.append(Self.colon)
      self.writeValue(entry.value)
    }
    self.buffer.append(Self.closeBrace)
  }

  mutating func writeString(_ value: String) {
    self.buffer.append(Self.quote)
    for byte in value.utf8 {
      switch byte {
      case Self.quote: self.buffer.append(contentsOf: #"\""#.utf8)
      case Self.backslash: self.buffer.append(contentsOf: #"\\"#.utf8)
      case Self.backspace: self.buffer.append(contentsOf: #"\b"#.utf8)
      case Self.formFeed: self.buffer.append(contentsOf: #"\f"#.utf8)
      case Self.newline: self.buffer.append(contentsOf: #"\n"#.utf8)
      case Self.carriageReturn: self.buffer.append(contentsOf: #"\r"#.utf8)
      case Self.tab: self.buffer.append(contentsOf: #"\t"#.utf8)
      default:
        if byte < Self.controlCharLimit {
          self.buffer.append(contentsOf: #"\u"#.utf8)
          self.buffer.append(contentsOf: Self.encodeHexDigits(of: byte, count: 4))
        } else {
          self.buffer.append(byte)
        }
      }
    }
    self.buffer.append(Self.quote)
  }

  private mutating func writeInt(_ value: Int) {
    self.buffer.append(contentsOf: String(value).utf8)
  }

  private mutating func writeDouble(_ value: Double) {
    self.buffer.append(contentsOf: (value.isFinite ? String(value) : "null").utf8)
  }

  private mutating func writeBool(_ value: Bool) {
    self.buffer.append(contentsOf: (value ? "true" : "false").utf8)
  }

  mutating func writeValue(_ value: EdgeToolsValue) {
    switch value {
    case .null:
      self.buffer.append(contentsOf: "null".utf8)
    case .boolean(let bool):
      self.writeBool(bool)
    case .integer(let int):
      self.writeInt(int)
    case .number(let number):
      self.writeDouble(number)
    case .string(let string):
      self.writeString(string)
    case .array(let array):
      self.writeValueArray(array)
    case .object(let object):
      self.writeObject(object)
    }
  }

  private mutating func writeValueArray(_ values: [EdgeToolsValue]) {
    self.buffer.append(Self.openBracket)
    for (offset, value) in values.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeValue(value)
    }
    self.buffer.append(Self.closeBracket)
  }

  private mutating func writeObject(_ object: OrderedDictionary<String, EdgeToolsValue>) {
    self.buffer.append(Self.openBrace)
    for (offset, entry) in object.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeString(entry.key)
      self.buffer.append(Self.colon)
      self.writeValue(entry.value)
    }
    self.buffer.append(Self.closeBrace)
  }

  private static func encodeHexDigits(of byte: UInt8, count: Int) -> [UInt8] {
    var result = [UInt8](repeating: Self.zero, count: count)
    var value = UInt16(byte)
    for index in stride(from: count - 1, through: 0, by: -1) {
      result[index] = Self.hexDigitTable[Int(value & 0xF)]
      value >>= 4
    }
    return result
  }
}
