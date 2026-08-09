@nonexhaustive
public enum EdgeToolsGenerationPart: Hashable, Sendable {
  case text(String)
  case reasoning(String)
  case toolCall(EdgeRawToolCall)

  public var text: String? {
    guard case .text(let text) = self else { return nil }
    return text
  }

  public var reasoning: String? {
    guard case .reasoning(let reasoning) = self else { return nil }
    return reasoning
  }

  public var toolCall: EdgeRawToolCall? {
    guard case .toolCall(let toolCall) = self else { return nil }
    return toolCall
  }
}
