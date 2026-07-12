import Foundation

// MARK: - NeedleToolCallParser

public struct NeedleToolCallParser: Sendable {
  private enum Phase {
    case outsideBlock
    case insideArray
  }

  private enum JSONStringState {
    case outsideString
    case insideString(isEscaping: Bool)
  }

  private static let opener = "tool_call"

  private var phase = Phase.outsideBlock
  private var buffer = ""
  private var hasSeenArrayOpen = false
  private var braceDepth = 0
  private var currentObjectStartOffset: Int?
  private var scanOffset: Int?
  private var jsonStringState = JSONStringState.outsideString

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.buffer.append(token.stringValue)
    return switch self.phase {
    case .outsideBlock: self.handleOutsideBlock()
    case .insideArray: self.handleInsideArray()
    }
  }

  private mutating func handleOutsideBlock() -> EdgeRawToolCall? {
    guard let range = self.buffer.range(of: Self.opener) else { return nil }
    self.buffer.removeSubrange(..<range.upperBound)
    self.phase = .insideArray
    return self.handleInsideArray()
  }

  private mutating func handleInsideArray() -> EdgeRawToolCall? {
    guard self.consumeArrayOpenBracketIfNeeded() else { return nil }

    var startIndex = self.buffer.startIndex
    let offset = self.scanOffset ?? 0
    var scanIndex = self.buffer.index(startIndex, offsetBy: offset)
    while scanIndex < self.buffer.endIndex {
      let character = self.buffer[scanIndex]
      let nextIndex = self.buffer.index(after: scanIndex)

      self.updateJSONStringState(for: character)

      guard self.shouldTreatAsBoundary(character) else {
        scanIndex = nextIndex
        continue
      }

      switch character {
      case "{":
        self.openBrace(at: scanIndex)
        scanIndex = nextIndex

      case "}":
        let result = self.closeBrace(at: nextIndex)
        if let call = result.call {
          return call
        }
        if result.closed {
          startIndex = self.buffer.startIndex
          scanIndex = startIndex
        } else {
          scanIndex = nextIndex
        }

      case "]":
        if self.braceDepth == 0 {
          self.buffer.removeSubrange(..<nextIndex)
          return nil
        }
        scanIndex = nextIndex

      default:
        break
      }
    }

    self.scanOffset = self.buffer.distance(from: startIndex, to: scanIndex)
    return nil
  }

  private mutating func updateJSONStringState(for character: Character) {
    switch (self.jsonStringState, character) {
    case (.outsideString, "\""):
      self.jsonStringState = .insideString(isEscaping: false)
    case (.insideString(let isEscaping), "\\"):
      self.jsonStringState = .insideString(isEscaping: !isEscaping)
    case (.insideString(let isEscaping), "\""):
      self.jsonStringState = isEscaping ? .insideString(isEscaping: false) : .outsideString
    case (.insideString, _):
      self.jsonStringState = .insideString(isEscaping: false)
    case (.outsideString, _):
      break
    }
  }

  private func shouldTreatAsBoundary(_ character: Character) -> Bool {
    if case .insideString = self.jsonStringState {
      return false
    }
    return character.isEdgeToolCallBoundary
  }

  private mutating func consumeArrayOpenBracketIfNeeded() -> Bool {
    guard !self.hasSeenArrayOpen else { return true }
    guard let bracketIndex = self.buffer.firstIndex(of: "[") else { return false }
    self.buffer.removeSubrange(..<bracketIndex)
    self.hasSeenArrayOpen = true
    self.scanOffset = 0
    return true
  }

  private mutating func openBrace(at index: String.Index) {
    if self.braceDepth == 0 {
      self.currentObjectStartOffset = self.buffer.distance(from: self.buffer.startIndex, to: index)
    }
    self.braceDepth += 1
  }

  private mutating func closeBrace(
    at nextIndex: String.Index
  ) -> (closed: Bool, call: EdgeRawToolCall?) {
    guard self.braceDepth > 0 else { return (false, nil) }
    self.braceDepth -= 1
    guard self.braceDepth == 0, let startOffset = self.currentObjectStartOffset else {
      return (false, nil)
    }

    let start = self.buffer.index(self.buffer.startIndex, offsetBy: startOffset)
    let objectString = String(self.buffer[start..<nextIndex])
    self.buffer.removeSubrange(..<nextIndex)
    self.currentObjectStartOffset = nil
    self.scanOffset = 0
    return (true, self.parse(objectString: objectString))
  }

  private func parse(objectString: String) -> EdgeRawToolCall? {
    guard let data = objectString.data(using: .utf8) else { return nil }
    return try? JSONDecoder().decode(EdgeRawToolCall.self, from: data)
  }
}

extension Character {
  fileprivate var isEdgeToolCallBoundary: Bool {
    ["{", "}", "]"].contains(self)
  }
}
