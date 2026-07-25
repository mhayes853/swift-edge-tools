import OrderedCollections

private struct JSONStringState: Hashable, Sendable {
  private var isInsideString = false
  private var isEscaping = false

  mutating func consume(_ byte: UInt8) -> Bool {
    guard self.isInsideString else {
      if byte == UInt8(ascii: "\"") {
        self.isInsideString = true
        return false
      }
      return true
    }

    if self.isEscaping {
      self.isEscaping = false
    } else if byte == UInt8(ascii: "\\") {
      self.isEscaping = true
    } else if byte == UInt8(ascii: "\"") {
      self.isInsideString = false
    }
    return false
  }
}

extension Array where Element == UInt8 {
  func firstRange(of needle: [UInt8], startingAt start: Int = 0) -> Range<Int>? {
    guard !needle.isEmpty, start <= self.count - needle.count else { return nil }
    for index in start...(self.count - needle.count) where self[index] == needle[0] {
      let end = index + needle.count
      if self[index..<end].elementsEqual(needle) {
        return index..<end
      }
    }
    return nil
  }

  func firstCompleteJSONObjectRange() -> Range<Int>? {
    guard self.first == UInt8(ascii: "{") else { return nil }
    var depth = 0
    var stringState = JSONStringState()

    for index in self.indices {
      let byte = self[index]
      guard stringState.consume(byte) else { continue }
      if byte == UInt8(ascii: "{") {
        depth += 1
      } else if byte == UInt8(ascii: "}") {
        depth -= 1
        if depth == 0 { return 0..<(index + 1) }
      }
      guard depth >= 0 else { return nil }
    }
    return nil
  }

  func firstRangeOutsideJSONString(of needle: [UInt8]) -> Range<Int>? {
    var index = 0
    var stringState = JSONStringState()

    while index < self.count {
      let byte = self[index]
      if stringState.consume(byte), self[index...].starts(with: needle) {
        return index..<(index + needle.count)
      }
      index += 1
    }
    return nil
  }

  func firstRange(of needle: [UInt8], outside marker: [UInt8]) -> Range<Int>? {
    var index = 0
    var isInsideMarker = false
    while index < self.count {
      if let markerRange = self.firstRange(of: marker, startingAt: index),
        markerRange.lowerBound == index
      {
        isInsideMarker.toggle()
        index = markerRange.upperBound
      } else if !isInsideMarker,
        let needleRange = self.firstRange(of: needle, startingAt: index),
        needleRange.lowerBound == index
      {
        return needleRange
      } else {
        index += 1
      }
    }
    return nil
  }

  mutating func removeLeadingASCIIWhitespaceAndCommas() {
    while let first = self.first {
      guard first == UInt8(ascii: ",") || first.isASCIIWhitespace else { return }
      self.removeFirst()
    }
  }

  mutating func retainPossiblePrefix(of marker: [UInt8]) {
    let maximumLength = Swift.min(self.count, marker.count - 1)
    let retainedLength =
      stride(from: maximumLength, through: 1, by: -1)
      .first { length in
        self.suffix(length).elementsEqual(marker.prefix(length))
      } ?? 0
    self = Array(self.suffix(retainedLength))
  }
}

// MARK: - IncrementalToolCallList

struct IncrementalToolCallList: Hashable, Sendable {
  private let opener: [UInt8]
  private var buffer = [UInt8]()
  private var isInsideBlock = false
  private var hasSeenListStart = false

  init(opener: String) {
    self.opener = Array(opener.utf8)
  }

  mutating func append(_ token: EdgeToolsToken) {
    self.buffer.append(contentsOf: token.stringValue.utf8)
  }

  mutating func nextItem(
    findRange: ([UInt8]) -> Range<Int>?
  ) -> [UInt8]? {
    while true {
      guard self.enterBlockIfNeeded() else { return nil }
      guard self.consumeListStartIfNeeded() else { return nil }
      self.buffer.removeLeadingASCIIWhitespaceAndCommas()

      if self.buffer.first == UInt8(ascii: "]") {
        self.buffer.removeFirst()
        self.isInsideBlock = false
        self.hasSeenListStart = false
        continue
      }

      guard let itemRange = findRange(self.buffer) else { return nil }
      let item = Array(self.buffer[itemRange])
      self.buffer.removeSubrange(..<itemRange.upperBound)
      return item
    }
  }

  private mutating func enterBlockIfNeeded() -> Bool {
    guard !self.isInsideBlock else { return true }
    guard let openerRange = self.buffer.firstRange(of: self.opener) else {
      self.buffer.retainPossiblePrefix(of: self.opener)
      return false
    }
    self.buffer.removeSubrange(..<openerRange.upperBound)
    self.isInsideBlock = true
    return true
  }

  private mutating func consumeListStartIfNeeded() -> Bool {
    guard !self.hasSeenListStart else { return true }
    guard let listStart = self.buffer.firstIndex(of: UInt8(ascii: "[")) else { return false }
    self.buffer.removeSubrange(...listStart)
    self.hasSeenListStart = true
    return true
  }
}

// MARK: - IncrementalToolCallBlock

struct IncrementalToolCallBlock: Hashable, Sendable {
  private let opener: [UInt8]
  private let closer: [UInt8]
  private var buffer = [UInt8]()
  private var isInsideBlock = false

  init(opener: String, closer: String) {
    self.opener = Array(opener.utf8)
    self.closer = Array(closer.utf8)
  }

  mutating func append(_ token: EdgeToolsToken) {
    self.buffer.append(contentsOf: token.stringValue.utf8)
  }

  mutating func nextPayload(respectingJSONStringBoundaries: Bool) -> [UInt8]? {
    let closer = self.closer
    return self.nextPayload { buffer in
      respectingJSONStringBoundaries
        ? buffer.firstRangeOutsideJSONString(of: closer)
        : buffer.firstRange(of: closer)
    }
  }

  mutating func nextPayload(outside marker: [UInt8]) -> [UInt8]? {
    let closer = self.closer
    return self.nextPayload { $0.firstRange(of: closer, outside: marker) }
  }

  private mutating func nextPayload(findCloser: ([UInt8]) -> Range<Int>?) -> [UInt8]? {
    if !self.isInsideBlock {
      guard let openerRange = self.buffer.firstRange(of: self.opener) else {
        self.buffer.retainPossiblePrefix(of: self.opener)
        return nil
      }
      self.buffer.removeSubrange(..<openerRange.upperBound)
      self.isInsideBlock = true
    }

    guard let closerRange = findCloser(self.buffer) else { return nil }
    let payload = Array(self.buffer[..<closerRange.lowerBound])
    self.buffer.removeSubrange(..<closerRange.upperBound)
    self.isInsideBlock = false
    return payload
  }
}

// MARK: - ToolCallStringCursor

struct ToolCallStringCursor: Hashable, Sendable {
  let source: String
  private(set) var index: String.Index

  var current: Character? {
    self.index < self.source.endIndex ? self.source[self.index] : nil
  }

  var remainder: Substring {
    self.source[self.index...]
  }

  var isAtEnd: Bool {
    self.index == self.source.endIndex
  }

  init(_ source: String) {
    self.source = source
    self.index = source.startIndex
  }

  mutating func advance() {
    self.index = self.source.index(after: self.index)
  }

  mutating func consume(character: Character) -> Bool {
    guard self.current == character else { return false }
    self.advance()
    return true
  }

  mutating func consume(_ value: String) -> Bool {
    guard self.remainder.hasPrefix(value) else { return false }
    self.index = self.source.index(self.index, offsetBy: value.count)
    return true
  }

  mutating func skipWhitespace() {
    while self.current?.isWhitespace == true {
      self.advance()
    }
  }

  mutating func read(while predicate: (Character) -> Bool) -> String {
    let start = self.index
    while let character = self.current, predicate(character) {
      self.advance()
    }
    return String(self.source[start..<self.index])
  }

  mutating func read(until delimiter: String) -> String? {
    guard let range = self.source[self.index...].firstRange(of: delimiter) else { return nil }
    let result = String(self.source[self.index..<range.lowerBound])
    self.index = range.lowerBound
    return result
  }
}

// MARK: - ToolCallValueReader

protocol ToolCallValueReader {
  var cursor: ToolCallStringCursor { get set }

  mutating func parseValue() -> EdgeToolsValue?
  mutating func parseObjectKey() -> String?
}

extension ToolCallValueReader {
  mutating func parseArray() -> EdgeToolsValue? {
    guard self.cursor.consume("[") else { return nil }
    self.cursor.skipWhitespace()
    var values = [EdgeToolsValue]()
    while !self.cursor.consume("]") {
      guard let value = self.parseValue() else { return nil }
      values.append(value)
      self.cursor.skipWhitespace()
      if self.cursor.consume("]") { break }
      guard self.cursor.consume(",") else { return nil }
      self.cursor.skipWhitespace()
    }
    return .array(values)
  }

  mutating func parseObjectBody() -> OrderedDictionary<String, EdgeToolsValue>? {
    var object = OrderedDictionary<String, EdgeToolsValue>()
    while !self.cursor.consume("}") {
      guard let key = self.parseObjectKey() else { return nil }
      self.cursor.skipWhitespace()
      guard let value = self.parseValue() else { return nil }
      object[key] = value
      self.cursor.skipWhitespace()
      if self.cursor.consume("}") { break }
      guard self.cursor.consume(",") else { return nil }
      self.cursor.skipWhitespace()
    }
    return object
  }
}
