#if MLX && Transformers && canImport(MLX)
  import Foundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import Tokenizers

  extension EdgeToolsLanguageModel where Self: LLMModel, Prompt == EdgeToolsLLMPrompt {
    public func process(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      using tokenizer: any EdgeToolsTokenizer
    ) throws -> sending LMInput {
      guard let tokenizer = tokenizer as? EdgeToolsPreTrainedTokenizer else {
        throw EdgeToolsMLXEngineError.unsupportedTokenizer
      }
      let preTrainedTokenizer = tokenizer.tokenizer
      let tokenIDs = try preTrainedTokenizer.applyChatTemplate(
        messages: prompt.mlxMessages(),
        tools: tools.mlxToolSpecs,
        additionalContext: nil
      )
      return LMInput(tokens: MLXArray(tokenIDs))
    }
  }

  extension EdgeToolsLLMPrompt {
    fileprivate func mlxMessages() throws -> [MLXLMCommon.Message] {
      let result = try self.messages.reduce(
        into: (
          messages: [MLXLMCommon.Message](),
          pendingToolNames: [String](),
          nextToolIndex: 0
        )
      ) { result, message in
        var mlxMessage = try message.mlxMessage()
        if message.role == .assistant {
          result.pendingToolNames = message.toolCalls?.map(\.name) ?? []
          result.nextToolIndex = 0
        } else if message.role == .tool,
          result.pendingToolNames.indices.contains(result.nextToolIndex)
        {
          mlxMessage["name"] = result.pendingToolNames[result.nextToolIndex]
          result.nextToolIndex += 1
        }
        result.messages.append(mlxMessage)
      }
      return result.messages
    }

  }

  extension EdgeToolsLLMPrompt.Message {
    package func mlxMessage() throws -> MLXLMCommon.Message {
      var message: MLXLMCommon.Message = ["role": self.role.rawValue]
      if let content = self.content {
        message["content"] = content
      }
      if let toolResponse = self.toolResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        message["content"] = String(decoding: try encoder.encode(toolResponse), as: UTF8.self)
      }
      if let toolCalls = self.toolCalls, !toolCalls.isEmpty {
        message["tool_calls"] = toolCalls.map(\.mlxToolCall)
      }
      return message
    }
  }

  extension EdgeRawToolCall {
    fileprivate var mlxToolCall: MLXLMCommon.Message {
      [
        "type": "function",
        "function": [
          "name": self.name,
          "arguments": self.arguments.mlxValue
        ] as MLXLMCommon.Message
      ]
    }
  }

  extension Sequence where Element == EdgeToolDefinition {
    package var mlxToolSpecs: [ToolSpec]? {
      let specifications = self.compactMap { definition -> ToolSpec? in
        guard definition.includesSchemaInInstructions else { return nil }
        return [
          "type": "function",
          "function": [
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.arguments.edgeToolsValue.mlxValue
          ] as MLXLMCommon.Message
        ]
      }
      return specifications.isEmpty ? nil : specifications
    }
  }

  extension EdgeToolsValue {
    fileprivate var mlxValue: any Sendable {
      switch self {
      case .array(let values): values.map(\.mlxValue)
      case .boolean(let value): value
      case .integer(let value): value
      case .null: NSNull()
      case .number(let value): value
      case .object(let object):
        Dictionary(uniqueKeysWithValues: object.map { ($0.key, $0.value.mlxValue) })
      case .string(let value): value
      }
    }
  }
#endif
