import Foundation

package func loadHuggingFaceBackendJSON(from tokenizerURL: URL) throws -> String {
  try huggingFaceBackendJSON(from: Data(contentsOf: tokenizerURL, options: .alwaysMapped))
}

package func huggingFaceBackendJSON(from data: Data) throws -> String {
  try data.withUnsafeBytes { bytes in
    var scanner = HuggingFaceBackendJSONScanner(buffer: bytes.bindMemory(to: UInt8.self))
    return "{\(try scanner.metadataFields().joined(separator: ","))}"
  }
}

private struct HuggingFaceBackendJSONScanner {
  private static let metadataKeys = ["decoder", "normalizer", "pre_tokenizer"]

  let buffer: UnsafeBufferPointer<UInt8>
  var index = 0

  mutating func metadataFields() throws -> [String] {
    self.skipWhitespace()
    try self.consume(0x7B)
    self.skipWhitespace()

    var values = [Range<Int>?](repeating: nil, count: Self.metadataKeys.count)
    while !self.consumeIfPresent(0x7D) {
      let key = try self.stringRange()
      self.skipWhitespace()
      try self.consume(0x3A)
      self.skipWhitespace()

      let value = try self.valueRange()
      if let metadataIndex = Self.metadataKeys.firstIndex(where: {
        self.buffer[key].elementsEqual("\"\($0)\"".utf8)
      }) {
        values[metadataIndex] = value
      }
      if values.allSatisfy({ $0 != nil }) { break }

      self.skipWhitespace()
      if self.consumeIfPresent(0x7D) { break }
      try self.consume(0x2C)
      self.skipWhitespace()
    }

    return try zip(Self.metadataKeys, values).compactMap { key, value in
      guard let value else { return nil }
      guard let jsonValue = String(bytes: self.buffer[value], encoding: .utf8) else {
        throw HuggingFaceBackendJSONError.invalidJSON
      }
      return "\"\(key)\":\(jsonValue)"
    }
  }

  private mutating func stringRange() throws -> Range<Int> {
    let start = self.index
    try self.consume(0x22)
    var escaped = false
    while self.index < self.buffer.count {
      let byte = self.buffer[self.index]
      self.index += 1
      if byte == 0x22, !escaped { return start..<self.index }
      escaped = byte == 0x5C && !escaped
    }
    throw HuggingFaceBackendJSONError.invalidJSON
  }

  private mutating func valueRange() throws -> Range<Int> {
    let start = self.index
    guard self.index < self.buffer.count else { throw HuggingFaceBackendJSONError.invalidJSON }

    switch self.buffer[self.index] {
    case 0x22: _ = try self.stringRange()
    case 0x7B, 0x5B: try self.container()
    default:
      while self.index < self.buffer.count, ![0x2C, 0x7D].contains(self.buffer[self.index]) {
        self.index += 1
      }
    }

    let end = self.buffer[start..<self.index].lastIndex(where: { !$0.isASCIIWhitespace })
      .map { $0 + 1 }
    guard let end, end > start else { throw HuggingFaceBackendJSONError.invalidJSON }
    return start..<end
  }

  private mutating func container() throws {
    var endings = [UInt8]()
    while self.index < self.buffer.count {
      let byte = self.buffer[self.index]
      if byte == 0x22 {
        _ = try self.stringRange()
        continue
      }

      self.index += 1
      switch byte {
      case 0x7B: endings.append(0x7D)
      case 0x5B: endings.append(0x5D)
      case 0x7D, 0x5D:
        guard endings.popLast() == byte else { throw HuggingFaceBackendJSONError.invalidJSON }
        if endings.isEmpty { return }
      default: break
      }
    }
    throw HuggingFaceBackendJSONError.invalidJSON
  }

  private mutating func skipWhitespace() {
    while self.index < self.buffer.count, self.buffer[self.index].isASCIIWhitespace {
      self.index += 1
    }
  }

  private mutating func consume(_ byte: UInt8) throws {
    guard self.consumeIfPresent(byte) else { throw HuggingFaceBackendJSONError.invalidJSON }
  }

  private mutating func consumeIfPresent(_ byte: UInt8) -> Bool {
    guard self.index < self.buffer.count, self.buffer[self.index] == byte else { return false }
    self.index += 1
    return true
  }

}

package struct HuggingFaceBackendJSONError: Hashable, Sendable, Error {
  package let message: String

  package static let invalidJSON = Self(message: "Invalid Hugging Face tokenizer JSON.")
}
