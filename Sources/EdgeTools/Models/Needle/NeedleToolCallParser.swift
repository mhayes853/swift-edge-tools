import Foundation

// MARK: - NeedleToolCallParser

public struct NeedleToolCallParser: EdgeToolCallParser, Sendable {
  private var list = IncrementalToolCallList(opener: "<tool_call>")

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.list.append(token)
    while let objectData = self.list.nextItem(findRange: { $0.firstCompleteJSONObjectRange() }) {
      if let call = try? JSONDecoder().decode(EdgeRawToolCall.self, from: objectData) {
        return call
      }
    }
    return nil
  }
}
