import Foundation
import OrderedCollections

extension EdgeToolDefinition {
  public func needleNormalized() -> Self {
    var definition = self
    definition.name = self.name.snakeCased()
    return definition
  }
}

extension EdgeToolDefinition {
  /// Encodes this tool definition as JSON data using the canonical Cactus
  /// Needle field order — the same order the Python training data generator
  /// uses (see `needle/dataset/generate.py` in cactus-compute/needle).
  ///
  /// Order for a tool definition:
  ///   `name, description, arguments`
  ///
  /// Order for a schema object:
  ///   1. `type` (if non-empty)
  ///   2. generic annotation keywords: `title, description, default, examples,
  ///      enum, const, allOf, anyOf, oneOf, not, if, then, else` (each omitted
  ///      when nil)
  ///   3. type-specific keys in JSON Schema order, branched on `type`:
  ///      - object: `properties, required, minProperties, maxProperties,
  ///        additionalProperties, patternProperties, propertyNames,
  ///        dependentRequired`
  ///      - array: `items, prefixItems, minItems, maxItems, uniqueItems,
  ///        contains, minContains, maxContains`
  ///      - integer/number: `multipleOf, minimum, exclusiveMinimum, maximum,
  ///        exclusiveMaximum`
  ///      - string: `minLength, maxLength, pattern`
  public func needlePromptEncoded() -> Data {
    var writer = JSONWriter()
    writer.writeToolDefinition(self)
    return Data(writer.buffer)
  }
}

extension EdgeToolsGenerationSchema {
  func needleGrammarEncoded() -> Data {
    var writer = JSONWriter()
    writer.writeSchema(self)
    return Data(writer.buffer)
  }
}

extension Sequence where Element == EdgeToolDefinition {
  /// Encodes a list of tool definitions as a JSON array using the canonical
  /// Cactus Needle field order. See ``encodedAsCanonicalJSON()`` for details.
  public func needlePromptEncoded() -> Data {
    var writer = JSONWriter()
    writer.writeArray { writer in
      for (offset, tool) in self.enumerated() {
        if offset > 0 { writer.writeComma() }
        writer.writeToolDefinition(tool)
      }
    }
    return Data(writer.buffer)
  }
}

// MARK: - JSONWriter

private struct JSONWriter {
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

  // MARK: - Top-level structure

  mutating func writeArray(_ body: (inout JSONWriter) -> Void) {
    self.buffer.append(Self.openBracket)
    body(&self)
    self.buffer.append(Self.closeBracket)
  }

  mutating func writeObject(_ body: (inout JSONWriter) -> Void) {
    self.buffer.append(Self.openBrace)
    body(&self)
    self.buffer.append(Self.closeBrace)
  }

  mutating func writeComma() {
    self.buffer.append(Self.comma)
  }

  mutating func writeColon() {
    self.buffer.append(Self.colon)
  }

  // MARK: - Tool definition

  mutating func writeToolDefinition(_ tool: EdgeToolDefinition) {
    self.writeObject { writer in
      writer.writeStringKey("name")
      writer.writeString(tool.name)
      writer.writeComma()
      writer.writeStringKey("description")
      writer.writeString(tool.description)
      writer.writeComma()
      writer.writeStringKey("arguments")
      writer.writeSchema(tool.arguments)
    }
  }

  // MARK: - Schema

  mutating func writeSchema(_ schema: EdgeToolsGenerationSchema) {
    switch schema {
    case .boolean(let value):
      self.writeBool(value)
    case .object(let object):
      self.writeSchemaObject(object)
    }
  }

  private mutating func writeSchemaObject(_ object: OrderedDictionary<EdgeToolsGenerationSchema.Key, EdgeToolsValue>) {
    self.buffer.append(Self.openBrace)
    for (offset, entry) in object.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeString(entry.key.rawValue)
      self.writeColon()
      self.writeValue(entry.value)
    }
    self.buffer.append(Self.closeBrace)
  }

  // MARK: - Primitives

  mutating func writeStringKey(_ key: String) {
    self.writeString(key)
    self.writeColon()
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

  mutating func writeInt(_ value: Int) {
    self.buffer.append(contentsOf: String(value).utf8)
  }

  mutating func writeDouble(_ value: Double) {
    if value.isFinite {
      self.buffer.append(contentsOf: String(value).utf8)
    } else {
      self.buffer.append(contentsOf: "null".utf8)
    }
  }

  mutating func writeBool(_ value: Bool) {
    self.buffer.append(contentsOf: (value ? "true" : "false").utf8)
  }

  mutating func writeStringArray(_ values: [String]) {
    self.buffer.append(Self.openBracket)
    for (offset, value) in values.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeString(value)
    }
    self.buffer.append(Self.closeBracket)
  }

  mutating func writeValueArray(_ values: [EdgeToolsValue]) {
    self.buffer.append(Self.openBracket)
    for (offset, value) in values.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeValue(value)
    }
    self.buffer.append(Self.closeBracket)
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
      self.buffer.append(Self.openBracket)
      for (offset, item) in array.enumerated() {
        if offset > 0 { self.buffer.append(Self.comma) }
        self.writeValue(item)
      }
      self.buffer.append(Self.closeBracket)
    case .object(let object):
      self.buffer.append(Self.openBrace)
      for (offset, entry) in object.enumerated() {
        if offset > 0 { self.buffer.append(Self.comma) }
        self.writeString(entry.key)
        self.writeColon()
        self.writeValue(entry.value)
      }
      self.buffer.append(Self.closeBrace)
    }
  }

  private static func encodeHexDigits(of byte: UInt8, count: Int) -> [UInt8] {
    var result: [UInt8] = Array(repeating: zero, count: count)
    var value = UInt16(byte)
    for index in stride(from: count - 1, through: 0, by: -1) {
      result[index] = hexDigitTable[Int(value & 0xF)]
      value >>= 4
    }
    return result
  }
}
