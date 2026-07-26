// MARK: - ProtobufReader

struct ProtobufReader {
  let bytes: [UInt8]
  var offset = 0

  var isAtEnd: Bool {
    self.offset == self.bytes.count
  }

  mutating func readTag() throws(NeedleSPTokenizerError) -> (field: Int, wire: Int) {
    let rawTag = try self.readVarint()
    let field = Int(rawTag >> 3)
    let wire = Int(rawTag & 0x07)
    guard field > 0 else {
      throw NeedleSPTokenizerError.invalidProtobuf(message: "contains an invalid tag.")
    }
    return (field, wire)
  }

  mutating func readVarint() throws(NeedleSPTokenizerError) -> UInt64 {
    var value: UInt64 = 0
    for index in 0..<10 {
      guard self.offset < self.bytes.count else {
        throw NeedleSPTokenizerError.invalidProtobuf(message: "is truncated.")
      }
      let byte = self.bytes[self.offset]
      self.offset += 1
      if index == 9 && byte > 1 {
        throw NeedleSPTokenizerError.invalidProtobuf(message: "contains an overflowing varint.")
      }
      value |= UInt64(byte & 0x7F) << UInt64(index * 7)
      if byte & 0x80 == 0 { return value }
    }
    throw NeedleSPTokenizerError.invalidProtobuf(message: "contains an overflowing varint.")
  }

  mutating func readFixed32() throws(NeedleSPTokenizerError) -> UInt32 {
    let bytes = try self.read(count: 4)
    return bytes.enumerated()
      .reduce(0) { result, element in
        result | UInt32(element.element) << UInt32(element.offset * 8)
      }
  }

  mutating func readLengthDelimited() throws(NeedleSPTokenizerError) -> [UInt8] {
    let length = try self.readVarint()
    guard length <= UInt64(Int.max) else {
      throw NeedleSPTokenizerError.invalidProtobuf(message: "contains a field that is too large.")
    }
    return try self.read(count: Int(length))
  }

  mutating func readString() throws(NeedleSPTokenizerError) -> String {
    let bytes = try self.readLengthDelimited()
    let string = String(decoding: bytes, as: UTF8.self)
    guard string.utf8.elementsEqual(bytes) else {
      throw NeedleSPTokenizerError.invalidProtobuf(message: "contains invalid UTF-8.")
    }
    return string
  }

  mutating func skip(wire: Int) throws(NeedleSPTokenizerError) {
    switch wire {
    case 0: _ = try self.readVarint()
    case 1: try self.skip(count: 8)
    case 2:
      let length = try self.readVarint()
      guard length <= UInt64(Int.max) else {
        throw NeedleSPTokenizerError.invalidProtobuf(message: "contains a field that is too large.")
      }
      try self.skip(count: Int(length))
    case 5: try self.skip(count: 4)
    default:
      throw NeedleSPTokenizerError.invalidProtobuf(
        message: "contains unsupported wire type \(wire)."
      )
    }
  }

  private mutating func skip(count: Int) throws(NeedleSPTokenizerError) {
    guard count >= 0, count <= self.bytes.count - self.offset else {
      throw NeedleSPTokenizerError.invalidProtobuf(message: "is truncated.")
    }
    self.offset += count
  }

  private mutating func read(count: Int) throws(NeedleSPTokenizerError) -> [UInt8] {
    guard count >= 0, count <= self.bytes.count - self.offset else {
      throw NeedleSPTokenizerError.invalidProtobuf(message: "is truncated.")
    }
    defer { self.offset += count }
    return Array(self.bytes[self.offset..<(self.offset + count)])
  }
}

// MARK: - ProtobufMessage

struct ProtobufMessage {
  let bytes: [UInt8]

  func lastBool(field: Int) throws -> Bool? {
    try self.lastVarint(field: field).map { $0 != 0 }
  }

  func lastInt32(field: Int) throws -> Int32? {
    try self.lastVarint(field: field).map { Int32(truncatingIfNeeded: $0) }
  }

  func lastFixed32(field: Int) throws -> UInt32? {
    try self.values(field: field, wire: 5) { try $0.readFixed32() }.last
  }

  func lastString(field: Int) throws -> String? {
    try self.values(field: field, wire: 2) { try $0.readString() }.last
  }

  func lastBytes(field: Int) throws -> [UInt8]? {
    try self.values(field: field, wire: 2) { try $0.readLengthDelimited() }.last
  }

  func lastMessage(field: Int) throws -> Self? {
    try self.messages(field: field).last
  }

  func messages(field: Int) throws -> [Self] {
    try self.values(field: field, wire: 2) {
      Self(bytes: try $0.readLengthDelimited())
    }
  }

  private func lastVarint(field: Int) throws -> UInt64? {
    try self.values(field: field, wire: 0) { try $0.readVarint() }.last
  }

  private func values<Value>(
    field: Int,
    wire: Int,
    decode: (inout ProtobufReader) throws -> Value
  ) throws -> [Value] {
    var reader = ProtobufReader(bytes: self.bytes)
    var values = [Value]()
    while !reader.isAtEnd {
      let tag = try reader.readTag()
      if tag.field == field, tag.wire == wire {
        values.append(try decode(&reader))
      } else {
        try reader.skip(wire: tag.wire)
      }
    }
    return values
  }
}
