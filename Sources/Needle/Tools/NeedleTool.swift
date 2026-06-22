import Foundation

// MARK: - NeedleTool

public protocol NeedleTool<Input, Output>: Sendable {
  associatedtype Input: ConvertibleFromNeedleValue & Sendable
  associatedtype Output: Sendable

  var name: String { get }
  var description: String { get }
  var arguments: NeedleGenerationSchema { get }

  func invoke(input: Input) async throws -> Output
}

extension NeedleTool where Input: NeedleGenerable {
  public var arguments: NeedleGenerationSchema {
    Input.needleGenerationSchema
  }
}

extension NeedleTool {
  public var definition: NeedleToolDefinition {
    NeedleToolDefinition(
      name: self.name,
      description: self.description,
      arguments: self.arguments
    )
  }
}

// MARK: - NeedleToolDefinition

public struct NeedleToolDefinition: Hashable, Sendable, Codable {
  public var name: String
  public var description: String
  public var arguments: NeedleGenerationSchema

  public init(name: String, description: String, arguments: NeedleGenerationSchema) {
    self.name = name
    self.description = description
    self.arguments = arguments
  }

  public func normalized() -> Self {
    var definition = self
    definition.name = self.name.snakeCased()
    return definition
  }
}

extension NeedleToolDefinition {
  /// Encodes this tool definition as JSON data using the canonical Cactus
  /// Needle field order — the same order the Python training data generator
  /// uses (see `needle/dataset/generate.py` in cactus-compute/needle).
  ///
  /// Order for a tool definition:
  ///   `name, description, arguments`
  ///
  /// Order for a schema object:
  ///   1. `type` (if non-empty)
  ///   2. generic annotation keywords: `title, description, default, readOnly,
  ///      writeOnly, examples, enum, const, allOf, anyOf, oneOf, not, if, then,
  ///      else, format` (each omitted when nil)
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

extension Sequence where Element == NeedleToolDefinition {
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

  mutating func writeToolDefinition(_ tool: NeedleToolDefinition) {
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

  mutating func writeSchema(_ schema: NeedleGenerationSchema) {
    switch schema {
    case .boolean(let value):
      self.buffer.append(contentsOf: (value ? "true" : "false").utf8)
    case .object(let object):
      self.writeSchemaObject(object)
    }
  }

  private mutating func writeSchemaObject(_ object: NeedleGenerationSchema.Object) {
    self.writeObject { writer in
      var isFirst = true
      func separator() {
        if isFirst { isFirst = false } else { writer.writeComma() }
      }

      if let type = object.type, !type.containedTypes.isEmpty {
        separator()
        writer.writeStringKey("type")
        writer.writeType(type)
      }
      if let title = object.title {
        separator()
        writer.writeStringKey("title")
        writer.writeString(title)
      }
      if let description = object.description {
        separator()
        writer.writeStringKey("description")
        writer.writeString(description)
      }
      if let defaultValue = object.default {
        separator()
        writer.writeStringKey("default")
        writer.writeValue(defaultValue)
      }
      if let readOnly = object.readOnly {
        separator()
        writer.writeStringKey("readOnly")
        writer.writeBool(readOnly)
      }
      if let writeOnly = object.writeOnly {
        separator()
        writer.writeStringKey("writeOnly")
        writer.writeBool(writeOnly)
      }
      if let examples = object.examples {
        separator()
        writer.writeStringKey("examples")
        writer.writeValueArray(examples)
      }
      if let enumValues = object.enum {
        separator()
        writer.writeStringKey("enum")
        writer.writeValueArray(enumValues)
      }
      if let const = object.const {
        separator()
        writer.writeStringKey("const")
        writer.writeValue(const)
      }
      if let allOf = object.allOf {
        separator()
        writer.writeStringKey("allOf")
        writer.writeSchemaArray(allOf)
      }
      if let anyOf = object.anyOf {
        separator()
        writer.writeStringKey("anyOf")
        writer.writeSchemaArray(anyOf)
      }
      if let oneOf = object.oneOf {
        separator()
        writer.writeStringKey("oneOf")
        writer.writeSchemaArray(oneOf)
      }
      if let not = object.not {
        separator()
        writer.writeStringKey("not")
        writer.writeSchema(not)
      }
      if let ifClause = object.if {
        separator()
        writer.writeStringKey("if")
        writer.writeSchema(ifClause)
      }
      if let thenClause = object.then {
        separator()
        writer.writeStringKey("then")
        writer.writeSchema(thenClause)
      }
      if let elseClause = object.`else` {
        separator()
        writer.writeStringKey("else")
        writer.writeSchema(elseClause)
      }
      if let format = object.format {
        separator()
        writer.writeStringKey("format")
        writer.writeString(format)
      }

      if let array = object.valueSchema?.array {
        if let items = array.items {
          separator()
          writer.writeStringKey("items")
          writer.writeSchema(items)
        }
        if let prefixItems = array.prefixItems {
          separator()
          writer.writeStringKey("prefixItems")
          writer.writeSchemaArray(prefixItems)
        }
        if let minItems = array.minItems {
          separator()
          writer.writeStringKey("minItems")
          writer.writeInt(minItems)
        }
        if let maxItems = array.maxItems {
          separator()
          writer.writeStringKey("maxItems")
          writer.writeInt(maxItems)
        }
        if let uniqueItems = array.uniqueItems {
          separator()
          writer.writeStringKey("uniqueItems")
          writer.writeBool(uniqueItems)
        }
        if let contains = array.contains {
          separator()
          writer.writeStringKey("contains")
          writer.writeSchema(contains)
        }
        if let minContains = array.minContains {
          separator()
          writer.writeStringKey("minContains")
          writer.writeInt(minContains)
        }
        if let maxContains = array.maxContains {
          separator()
          writer.writeStringKey("maxContains")
          writer.writeInt(maxContains)
        }
      }
      if let integer = object.valueSchema?.integer {
        if let multipleOf = integer.multipleOf {
          separator()
          writer.writeStringKey("multipleOf")
          writer.writeInt(multipleOf)
        }
        if let minimum = integer.minimum {
          separator()
          writer.writeStringKey("minimum")
          writer.writeInt(minimum)
        }
        if let exclusiveMinimum = integer.exclusiveMinimum {
          separator()
          writer.writeStringKey("exclusiveMinimum")
          writer.writeInt(exclusiveMinimum)
        }
        if let maximum = integer.maximum {
          separator()
          writer.writeStringKey("maximum")
          writer.writeInt(maximum)
        }
        if let exclusiveMaximum = integer.exclusiveMaximum {
          separator()
          writer.writeStringKey("exclusiveMaximum")
          writer.writeInt(exclusiveMaximum)
        }
      }
      if let number = object.valueSchema?.number {
        if let multipleOf = number.multipleOf {
          separator()
          writer.writeStringKey("multipleOf")
          writer.writeDouble(multipleOf)
        }
        if let minimum = number.minimum {
          separator()
          writer.writeStringKey("minimum")
          writer.writeDouble(minimum)
        }
        if let exclusiveMinimum = number.exclusiveMinimum {
          separator()
          writer.writeStringKey("exclusiveMinimum")
          writer.writeDouble(exclusiveMinimum)
        }
        if let maximum = number.maximum {
          separator()
          writer.writeStringKey("maximum")
          writer.writeDouble(maximum)
        }
        if let exclusiveMaximum = number.exclusiveMaximum {
          separator()
          writer.writeStringKey("exclusiveMaximum")
          writer.writeDouble(exclusiveMaximum)
        }
      }
      if let string = object.valueSchema?.string {
        if let minLength = string.minLength {
          separator()
          writer.writeStringKey("minLength")
          writer.writeInt(minLength)
        }
        if let maxLength = string.maxLength {
          separator()
          writer.writeStringKey("maxLength")
          writer.writeInt(maxLength)
        }
        if let pattern = string.pattern {
          separator()
          writer.writeStringKey("pattern")
          writer.writeString(pattern)
        }
      }
      if let obj = object.valueSchema?.object {
        if let properties = obj.properties, !properties.isEmpty {
          separator()
          writer.writeStringKey("properties")
          writer.writeObject { pw in
            for (offset, key) in properties.keys.sorted().enumerated() {
              if offset > 0 { pw.writeComma() }
              pw.writeStringKey(key)
              pw.writeSchema(properties[key] ?? .boolean(true))
            }
          }
        }
        if let required = obj.required {
          separator()
          writer.writeStringKey("required")
          writer.writeStringArray(required)
        }
        if let minProperties = obj.minProperties {
          separator()
          writer.writeStringKey("minProperties")
          writer.writeInt(minProperties)
        }
        if let maxProperties = obj.maxProperties {
          separator()
          writer.writeStringKey("maxProperties")
          writer.writeInt(maxProperties)
        }
        if let additionalProperties = obj.additionalProperties {
          separator()
          writer.writeStringKey("additionalProperties")
          writer.writeSchema(additionalProperties)
        }
        if let patternProperties = obj.patternProperties, !patternProperties.isEmpty {
          separator()
          writer.writeStringKey("patternProperties")
          writer.writeObject { pw in
            for (offset, key) in patternProperties.keys.sorted().enumerated() {
              if offset > 0 { pw.writeComma() }
              pw.writeStringKey(key)
              pw.writeSchema(patternProperties[key] ?? .boolean(true))
            }
          }
        }
        if let propertyNames = obj.propertyNames {
          separator()
          writer.writeStringKey("propertyNames")
          writer.writeSchema(propertyNames)
        }
        if let dependentRequired = obj.dependentRequired, !dependentRequired.isEmpty {
          separator()
          writer.writeStringKey("dependentRequired")
          writer.writeObject { pw in
            for (offset, key) in dependentRequired.keys.sorted().enumerated() {
              if offset > 0 { pw.writeComma() }
              pw.writeStringKey(key)
              pw.writeStringArray(dependentRequired[key] ?? [])
            }
          }
        }
      }
    }
  }

  private mutating func writeSchemaArray(_ schemas: [NeedleGenerationSchema]) {
    self.buffer.append(Self.openBracket)
    for (offset, schema) in schemas.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeSchema(schema)
    }
    self.buffer.append(Self.closeBracket)
  }

  private mutating func writeType(_ type: NeedleGenerationSchema.ValueType) {
    let contained = type.containedTypes
    if contained.count == 1 {
      self.writeString(contained[0].canonicalName)
    } else {
      self.buffer.append(Self.openBracket)
      for (offset, member) in contained.enumerated() {
        if offset > 0 { self.buffer.append(Self.comma) }
        self.writeString(member.canonicalName)
      }
      self.buffer.append(Self.closeBracket)
    }
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
      case Self.quote: self.buffer.append(contentsOf: #""#.utf8)
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

  mutating func writeValueArray(_ values: [NeedleValue]) {
    self.buffer.append(Self.openBracket)
    for (offset, value) in values.enumerated() {
      if offset > 0 { self.buffer.append(Self.comma) }
      self.writeValue(value)
    }
    self.buffer.append(Self.closeBracket)
  }

  mutating func writeValue(_ value: NeedleValue) {
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
      for (offset, key) in object.keys.sorted().enumerated() {
        if offset > 0 { self.buffer.append(Self.comma) }
        self.writeString(key)
        self.writeColon()
        self.writeValue(object[key] ?? .null)
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

extension NeedleGenerationSchema.ValueType {
  fileprivate var canonicalName: String {
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

  fileprivate var containedTypes: [Self] {
    let allTypes = [Self.integer, .string, .boolean, .array, .object, .number, .null]
    return allTypes.filter { self.contains($0) }
  }
}
