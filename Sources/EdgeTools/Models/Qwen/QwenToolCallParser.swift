import Foundation
import OrderedCollections

// MARK: - QwenJSONToolCallParser

public struct QwenJSONToolCallParser: EdgeToolCallParser, Sendable {
  private var block = IncrementalToolCallBlock(
    opener: "<tool_call>",
    closer: "</tool_call>"
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.block.append(token)
    while let payload = self.block.nextPayload(respectingJSONStringBoundaries: true) {
      if let call = try? JSONDecoder().decode(EdgeRawToolCall.self, from: payload) {
        return call
      }
    }
    return nil
  }
}

// MARK: - QwenXMLToolCallParser

public struct QwenXMLToolCallParser: EdgeToolCallParser, Sendable {
  private var block = IncrementalToolCallBlock(
    opener: "<tool_call>",
    closer: "</tool_call>"
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.block.append(token)
    while let payloadData = self.block.nextPayload(respectingJSONStringBoundaries: false) {
      guard let payload = String(data: payloadData, encoding: .utf8) else { continue }
      if let call = Self.parse(payload) {
        return call
      }
    }
    return nil
  }

  private static func parse(_ payload: String) -> EdgeRawToolCall? {
    guard let functionStart = payload.range(of: "<function=") else { return nil }
    guard let nameEnd = payload[functionStart.upperBound...].firstIndex(of: ">") else { return nil }
    guard let functionEnd = payload.range(of: "</function>", range: nameEnd..<payload.endIndex)
    else {
      return nil
    }

    let name = payload[functionStart.upperBound..<nameEnd]
      .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !name.isEmpty else { return nil }

    let bodyStart = payload.index(after: nameEnd)
    let body = payload[bodyStart..<functionEnd.lowerBound]
    guard let arguments = Self.parseParameters(body) else { return nil }
    return EdgeRawToolCall(name: name, arguments: .object(arguments))
  }

  private static func parseParameters(
    _ body: Substring
  ) -> OrderedDictionary<String, EdgeToolsValue>? {
    var arguments = OrderedDictionary<String, EdgeToolsValue>()
    var searchStart = body.startIndex

    while let parameterStart = body.range(of: "<parameter=", range: searchStart..<body.endIndex) {
      guard let nameEnd = body[parameterStart.upperBound...].firstIndex(of: ">") else { return nil }
      let valueStart = body.index(after: nameEnd)
      guard
        let parameterEnd = body.range(
          of: "</parameter>",
          range: valueStart..<body.endIndex
        )
      else { return nil }

      let name = body[parameterStart.upperBound..<nameEnd]
        .trimmingCharacters(in: .whitespacesAndNewlines)
      guard !name.isEmpty else { return nil }
      let source = body[valueStart..<parameterEnd.lowerBound]
        .trimmingCharacters(in: .whitespacesAndNewlines)
      arguments[name] = Self.parseParameterValue(source)
      searchStart = parameterEnd.upperBound
    }
    return arguments
  }

  private static func parseParameterValue(_ source: String) -> EdgeToolsValue {
    switch source {
    case "True": return true
    case "False": return false
    case "None": return nil
    default:
      guard let data = source.data(using: .utf8) else { return .string(source) }
      return (try? JSONDecoder().decode(EdgeToolsValue.self, from: data)) ?? .string(source)
    }
  }
}

// MARK: - Qwen

public typealias Qwen3ToolCallParser = QwenJSONToolCallParser
public typealias Qwen3P5ToolCallParser = QwenXMLToolCallParser
public typealias Qwen3P6ToolCallParser = QwenXMLToolCallParser
