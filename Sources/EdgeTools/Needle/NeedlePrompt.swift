import Foundation
import OrderedCollections

// MARK: - NeedlePrompt

public struct NeedlePrompt: EdgeToolsPrompt {
  public var system: String
  public var user: String
  public var tools: [any EdgeTool]

  public init(system: String, user: String, tools: [any EdgeTool]) {
    self.system = system
    self.user = user
    self.tools = tools
  }
}

// MARK: - Formatting

extension NeedlePrompt {
  public func formatted() throws -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema = try self.tools.map(\.definition).map { $0.needleNormalized() }.needlePromptEncoded()
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }

  public func tokenized(
    using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
  ) throws -> [EdgeToolsToken] {
    let tokenIds = tokenizer.encode(text: try self.formatted())
    let tokens = tokenizer.convertIdsToTokens(tokenIds)
    return zip(tokenIds, tokens)
      .compactMap { (tokenId, token) in
        token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
      }
  }
}

// MARK: - EdgeToolDefinition

extension EdgeToolDefinition {
  public func needleNormalized() -> Self {
    var definition = self
    definition.name = self.name.snakeCased()
    return definition
  }

  fileprivate func needlePromptEncoded() throws -> String {
    let name = String(decoding: try JSONEncoder().encode(self.name), as: UTF8.self)
    let description = String(decoding: try JSONEncoder().encode(self.description), as: UTF8.self)
    let arguments = try self.arguments.needleGrammarEncoded()
    return #"{"name":\#(name),"description":\#(description),"arguments":\#(arguments)}"#
  }
}

extension Sequence where Element == EdgeToolDefinition {
  fileprivate func needlePromptEncoded() throws -> String {
    let definitions = try self.map { try $0.needlePromptEncoded() }
    return "[\(definitions.joined(separator: ","))]"
  }
}

// MARK: - Generation Schema

extension EdgeToolsGenerationSchema {
  func needleGrammarEncoded() throws -> String {
    var writer = JSONWriter()
    writer.writeSchema(self)
    return String(decoding: writer.buffer, as: UTF8.self)
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
        self.buffer.append(Self.colon)
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
