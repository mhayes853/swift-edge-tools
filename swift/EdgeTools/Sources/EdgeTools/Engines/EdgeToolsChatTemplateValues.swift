#if FoundationEssentials
  import EdgeToolsCore
  import OrderedCollections
  import _EdgeToolsFoundation

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
          "content": .string(String(decoding: try Self.encode(message.response), as: UTF8.self)),
          "name": .string(message.name)
        ]
      }
    }

    private func assistantChatTemplateValue(
      parts: [EdgeToolsGenerationPart]
    ) -> EdgeToolsValue {
      var message: OrderedDictionary<String, EdgeToolsValue> = ["role": "assistant"]
      let content = parts.compactMap(\.text).joined()
      let reasoning = parts.compactMap(\.reasoning).joined()
      let toolCalls = parts.compactMap(\.toolCall)
      if !content.isEmpty {
        message["content"] = .string(content)
      }
      if !reasoning.isEmpty {
        message["reasoning_content"] = .string(reasoning)
        message["thinking"] = .string(reasoning)
      }
      if !toolCalls.isEmpty {
        message["tool_calls"] = .array(toolCalls.map(\.chatTemplateValue))
      }
      return .object(message)
    }

    private static func encode(_ value: EdgeToolsValue) throws -> Data {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
      return try encoder.encode(value)
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
      let values = self.filter(\.includesSchemaInInstructions).map(\.chatTemplateValue)
      return values.isEmpty ? nil : values
    }
  }
#endif
