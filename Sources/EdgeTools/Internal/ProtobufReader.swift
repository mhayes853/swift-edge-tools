// MARK: - ProtobufReader

struct ProtobufReader {
  let bytes: [UInt8]
  var offset = 0

  var isAtEnd: Bool {
    self.offset == self.bytes.count
  }

  mutating func readTag() throws(ProtobufReaderError) -> (field: Int, wire: Int) {
    let rawTag = try self.readVarint()
    let field = Int(rawTag >> 3)
    let wire = Int(rawTag & 0x07)
    guard field > 0 else { throw ProtobufReaderError.invalidTag }
    return (field, wire)
  }

  mutating func readVarint() throws(ProtobufReaderError) -> UInt64 {
    var value: UInt64 = 0
    for index in 0..<10 {
      guard self.offset < self.bytes.count else { throw ProtobufReaderError.truncated }
      let byte = self.bytes[self.offset]
      self.offset += 1
      if index == 9 && byte > 1 {
        throw ProtobufReaderError.overflowingVarint
      }
      value |= UInt64(byte & 0x7F) << UInt64(index * 7)
      if byte & 0x80 == 0 { return value }
    }
    throw ProtobufReaderError.overflowingVarint
  }

  mutating func readBool() throws(ProtobufReaderError) -> Bool {
    try self.readVarint() != 0
  }

  mutating func readFixed32() throws(ProtobufReaderError) -> UInt32 {
    let bytes = try self.read(count: 4)
    return bytes.enumerated()
      .reduce(0) { result, element in
        result | UInt32(element.element) << UInt32(element.offset * 8)
      }
  }

  mutating func readLengthDelimited() throws(ProtobufReaderError) -> [UInt8] {
    let length = try self.readVarint()
    guard length <= UInt64(Int.max) else { throw ProtobufReaderError.fieldTooLarge }
    return try self.read(count: Int(length))
  }

  mutating func readString() throws(ProtobufReaderError) -> String {
    let bytes = try self.readLengthDelimited()
    let string = String(decoding: bytes, as: UTF8.self)
    guard string.utf8.elementsEqual(bytes) else { throw ProtobufReaderError.invalidUTF8 }
    return string
  }

  mutating func skip(wire: Int) throws(ProtobufReaderError) {
    switch wire {
    case 0: _ = try self.readVarint()
    case 1: _ = try self.read(count: 8)
    case 2: _ = try self.readLengthDelimited()
    case 5: _ = try self.read(count: 4)
    default: throw ProtobufReaderError.unsupportedWireType(wire)
    }
  }

  private mutating func read(count: Int) throws(ProtobufReaderError) -> [UInt8] {
    guard count >= 0, count <= self.bytes.count - self.offset else {
      throw ProtobufReaderError.truncated
    }
    defer { self.offset += count }
    return Array(self.bytes[self.offset..<(self.offset + count)])
  }
}

// MARK: - ProtobufReaderError

enum ProtobufReaderError: Hashable, Sendable, Error {
  case invalidTag
  case truncated
  case overflowingVarint
  case fieldTooLarge
  case invalidUTF8
  case unsupportedWireType(Int)

  var message: String {
    switch self {
    case .invalidTag: "contains an invalid tag."
    case .truncated: "is truncated."
    case .overflowingVarint: "contains an overflowing varint."
    case .fieldTooLarge: "contains a field that is too large."
    case .invalidUTF8: "contains invalid UTF-8."
    case .unsupportedWireType(let wire): "contains unsupported wire type \(wire)."
    }
  }
}
