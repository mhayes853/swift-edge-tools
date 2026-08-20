import EdgeToolsCore
import OrderedCollections

// MARK: - Transcript Conversion

extension EdgeToolsTranscript {
  func chatTemplateMessages(
    userMessage: (UserMessage) throws -> EdgeToolsValue = {
      ["role": "user", "content": .string($0.content)]
    }
  ) throws -> [EdgeToolsValue] {
    try self.messages.map { message in
      guard case .user(let message) = message else {
        return try message.chatTemplateValue()
      }
      return try userMessage(message)
    }
  }
}

extension EdgeToolsTranscript.Message {
  public func chatTemplateValue() throws -> EdgeToolsValue {
    switch self {
    case .system(let message):
      ["role": "system", "content": .string(message.content)]
    case .user(let message):
      ["role": "user", "content": .string(message.content)]
    case .assistant(let message):
      self.assistantChatTemplateValue(parts: message.parts)
    case .tool(let message):
      [
        "role": "tool",
        "content": .string(message.response.orderedJSONString()),
        "name": .string(message.name)
      ]
    }
  }

  private func assistantChatTemplateValue(
    parts: [EdgeToolsGenerationPart]
  ) -> EdgeToolsValue {
    var message: OrderedDictionary<String, EdgeToolsValue> = ["role": "assistant"]
    let content = parts.compactMap { $0.text }.joined()
    let reasoning = parts.compactMap { $0.reasoning }.joined()
    let toolCalls = parts.compactMap { $0.toolCall }
    if !content.isEmpty {
      message["content"] = .string(content)
    }
    if !reasoning.isEmpty {
      message["reasoning_content"] = .string(reasoning)
      message["thinking"] = .string(reasoning)
    }
    if !toolCalls.isEmpty {
      message["tool_calls"] = .array(toolCalls.map { $0.chatTemplateValue })
    }
    return .object(message)
  }
}

// MARK: - Tool Conversion

extension EdgeRawToolCall {
  var chatTemplateValue: EdgeToolsValue {
    [
      "type": "function",
      "function": [
        "name": .string(self.name),
        "arguments": self.arguments
      ]
    ]
  }
}

extension EdgeToolDefinition {
  public var chatTemplateValue: EdgeToolsValue {
    [
      "type": "function",
      "function": [
        "name": .string(self.name),
        "description": .string(self.description),
        "parameters": self.arguments.edgeToolsValue
      ]
    ]
  }
}

extension Sequence where Element == EdgeToolDefinition {
  var chatTemplateToolValues: [EdgeToolsValue]? {
    let values = self
      .filter { $0.includesSchemaInInstructions }
      .map { $0.chatTemplateValue }
    return values.isEmpty ? nil : values
  }
}
