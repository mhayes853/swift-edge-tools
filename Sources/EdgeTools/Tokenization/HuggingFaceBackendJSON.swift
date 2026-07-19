import Foundation

package func loadHuggingFaceBackendJSON(from tokenizerURL: URL) throws -> String {
  let data = try Data(contentsOf: tokenizerURL, options: .alwaysMapped)
  return try huggingFaceBackendJSON(from: data)
}

package func huggingFaceBackendJSON(from data: Data) throws -> String {
  try data.withUnsafeBytes { bytes in
    let buffer = bytes.bindMemory(to: UInt8.self)
    var scanner = JSONTopLevelScanner(buffer: buffer)
    let values = try scanner.values(for: ["decoder", "normalizer", "pre_tokenizer"])
    let fields = ["decoder", "normalizer", "pre_tokenizer"]
      .compactMap { key -> String? in
        guard let range = values[key] else { return nil }
        let value = String(decoding: buffer[range], as: UTF8.self)
        return "\(SelfEncodedJSON.string(key)):\(value)"
      }
    return "{\(fields.joined(separator: ","))}"
  }
}

private enum SelfEncodedJSON {
  static func string(_ value: String) -> String {
    var result = "\""
    for scalar in value.unicodeScalars {
      switch scalar.value {
      case 0x22: result += "\\\""
      case 0x5C: result += "\\\\"
      case 0x00...0x1F: result += String(format: "\\u%04X", scalar.value)
      default: result.unicodeScalars.append(scalar)
      }
    }
    result += "\""
    return result
  }
}

private struct JSONTopLevelScanner {
  let buffer: UnsafeBufferPointer<UInt8>
  var index = 0

  mutating func values(for selectedKeys: Set<String>) throws -> [String: Range<Int>] {
    self.skipWhitespace()
    try self.consume(0x7B)
    self.skipWhitespace()

    var values = [String: Range<Int>]()
    guard !self.consumeIfPresent(0x7D) else { return values }

    while true {
      let key = try self.readString()
      self.skipWhitespace()
      try self.consume(0x3A)
      self.skipWhitespace()
      let valueRange = try self.readValueRange()
      if selectedKeys.contains(key) {
        values[key] = valueRange
      }
      if values.count == selectedKeys.count {
        return values
      }

      self.skipWhitespace()
      if self.consumeIfPresent(0x7D) {
        return values
      }
      try self.consume(0x2C)
      self.skipWhitespace()
    }
  }

  private mutating func readString() throws -> String {
    let range = try self.readStringRange()
    let data = Data(self.buffer[range])
    return try JSONDecoder().decode(String.self, from: data)
  }

  private mutating func readStringRange() throws -> Range<Int> {
    let start = self.index
    try self.consume(0x22)
    var escaped = false
    while self.index < self.buffer.count {
      let byte = self.buffer[self.index]
      self.index += 1
      if escaped {
        escaped = false
      } else if byte == 0x5C {
        escaped = true
      } else if byte == 0x22 {
        return start..<self.index
      }
    }
    throw HuggingFaceBackendJSONError.invalidJSON
  }

  private mutating func readValueRange() throws -> Range<Int> {
    let start = self.index
    guard self.index < self.buffer.count else {
      throw HuggingFaceBackendJSONError.invalidJSON
    }

    switch self.buffer[self.index] {
    case 0x22:
      _ = try self.readStringRange()
    case 0x7B, 0x5B:
      try self.skipContainer()
    default:
      while self.index < self.buffer.count {
        let byte = self.buffer[self.index]
        guard byte != 0x2C && byte != 0x7D else { break }
        self.index += 1
      }
    }

    var end = self.index
    while end > start && Self.isWhitespace(self.buffer[end - 1]) {
      end -= 1
    }
    guard end > start else { throw HuggingFaceBackendJSONError.invalidJSON }
    return start..<end
  }

  private mutating func skipContainer() throws {
    var delimiters = [UInt8]()
    while self.index < self.buffer.count {
      let byte = self.buffer[self.index]
      if byte == 0x22 {
        _ = try self.readStringRange()
        continue
      }
      self.index += 1
      switch byte {
      case 0x7B: delimiters.append(0x7D)
      case 0x5B: delimiters.append(0x5D)
      case 0x7D, 0x5D:
        guard delimiters.popLast() == byte else {
          throw HuggingFaceBackendJSONError.invalidJSON
        }
        if delimiters.isEmpty { return }
      default: break
      }
    }
    throw HuggingFaceBackendJSONError.invalidJSON
  }

  private mutating func skipWhitespace() {
    while self.index < self.buffer.count && Self.isWhitespace(self.buffer[self.index]) {
      self.index += 1
    }
  }

  private mutating func consume(_ byte: UInt8) throws {
    guard self.consumeIfPresent(byte) else {
      throw HuggingFaceBackendJSONError.invalidJSON
    }
  }

  private mutating func consumeIfPresent(_ byte: UInt8) -> Bool {
    guard self.index < self.buffer.count, self.buffer[self.index] == byte else { return false }
    self.index += 1
    return true
  }

  private static func isWhitespace(_ byte: UInt8) -> Bool {
    byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09
  }
}

package struct HuggingFaceBackendJSONError: Hashable, Sendable, Error {
  package let message: String

  package static let invalidJSON = Self(message: "Invalid Hugging Face tokenizer JSON.")
}
