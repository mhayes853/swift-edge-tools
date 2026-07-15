import Foundation

// MARK: - GemmaToolCallSyntax

struct GemmaToolCallSyntax: Hashable, Sendable {
  let opener: [UInt8]
  let closer: [UInt8]
  let stringMarker: String
  let decodesMarkedValues: Bool

  init(opener: String, closer: String, stringMarker: String, decodesMarkedValues: Bool) {
    self.opener = Array(opener.utf8)
    self.closer = Array(closer.utf8)
    self.stringMarker = stringMarker
    self.decodesMarkedValues = decodesMarkedValues
  }
}

// MARK: - GemmaToolCallParserCore

struct GemmaToolCallParserCore: Sendable {
  private let syntax: GemmaToolCallSyntax
  private var buffer = [UInt8]()
  private var isInsideCall = false

  init(syntax: GemmaToolCallSyntax) {
    self.syntax = syntax
  }

  mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.buffer.append(contentsOf: token.stringValue.utf8)

    while true {
      if !self.isInsideCall {
        guard let openerRange = self.buffer.firstRange(of: self.syntax.opener) else {
          self.buffer.retainPossiblePrefix(of: self.syntax.opener)
          return nil
        }
        self.buffer.removeSubrange(..<openerRange.upperBound)
        self.isInsideCall = true
      }

      let stringMarker = Array(self.syntax.stringMarker.utf8)
      guard
        let closerRange = self.buffer.firstRange(
          of: self.syntax.closer,
          outside: stringMarker
        )
      else { return nil }

      let payloadData = Data(self.buffer[..<closerRange.lowerBound])
      self.buffer.removeSubrange(..<closerRange.upperBound)
      self.isInsideCall = false
      guard let payload = String(data: payloadData, encoding: .utf8) else { continue }
      var reader = GemmaCallReader(source: payload, syntax: self.syntax)
      if let call = reader.parse() {
        return call
      }
    }
  }
}

extension Array where Element == UInt8 {
  fileprivate func firstRange(of needle: [UInt8], outside marker: [UInt8]) -> Range<Int>? {
    var index = 0
    var isInsideMarkedValue = false
    while index < self.count {
      if let markerRange = self.firstRange(of: marker, startingAt: index),
        markerRange.lowerBound == index
      {
        isInsideMarkedValue.toggle()
        index = markerRange.upperBound
        continue
      }
      if !isInsideMarkedValue,
        let needleRange = self.firstRange(of: needle, startingAt: index),
        needleRange.lowerBound == index
      {
        return needleRange
      }
      index += 1
    }
    return nil
  }
}

// MARK: - GemmaCallReader

private struct GemmaCallReader: ToolCallValueReader {
  let syntax: GemmaToolCallSyntax
  var cursor: ToolCallStringCursor

  init(source: String, syntax: GemmaToolCallSyntax) {
    self.syntax = syntax
    self.cursor = ToolCallStringCursor(source)
  }

  mutating func parse() -> EdgeRawToolCall? {
    self.cursor.skipWhitespace()
    guard self.cursor.consume("call:") else { return nil }
    guard let name = self.cursor.read(until: "{")?.trimmingCharacters(in: .whitespacesAndNewlines),
      !name.isEmpty,
      self.cursor.consume("{")
    else { return nil }

    self.cursor.skipWhitespace()
    guard let arguments = self.parseObjectBody() else { return nil }
    self.cursor.skipWhitespace()
    guard self.cursor.isAtEnd else { return nil }
    return EdgeRawToolCall(name: name, arguments: .object(arguments))
  }

  mutating func parseObjectKey() -> String? {
    guard
      let key = self.cursor.read(until: ":")?.trimmingCharacters(in: .whitespacesAndNewlines),
      !key.isEmpty,
      self.cursor.consume(":")
    else { return nil }
    return key
  }

  mutating func parseValue() -> EdgeToolsValue? {
    self.cursor.skipWhitespace()
    if self.cursor.remainder.hasPrefix(self.syntax.stringMarker) {
      return self.parseMarkedValue()
    }
    guard let character = self.cursor.current else { return nil }
    switch character {
    case "{":
      self.cursor.advance()
      return self.parseObjectBody().map(EdgeToolsValue.object)
    case "[": return self.parseArray()
    default:
      let token = self.parseBareValue()
      return switch token {
      case "true", "True": true
      case "false", "False": false
      case "null", "None": .null
      default:
        if token.isEmpty {
          nil
        } else if let integer = Int(token) {
          .integer(integer)
        } else if let number = Double(token) {
          .number(number)
        } else {
          .string(token)
        }
      }
    }
  }

  private mutating func parseMarkedValue() -> EdgeToolsValue? {
    guard self.cursor.consume(self.syntax.stringMarker) else { return nil }
    guard let value = self.cursor.read(until: self.syntax.stringMarker) else { return nil }
    guard self.cursor.consume(self.syntax.stringMarker) else { return nil }
    guard self.syntax.decodesMarkedValues, let data = value.data(using: .utf8) else {
      return .string(value)
    }
    return (try? JSONDecoder().decode(EdgeToolsValue.self, from: data)) ?? .string(value)
  }

  private mutating func parseBareValue() -> String {
    self.cursor.read { ![",", "}", "]"].contains($0) }
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
