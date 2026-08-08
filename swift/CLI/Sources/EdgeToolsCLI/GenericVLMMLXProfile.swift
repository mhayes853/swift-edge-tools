import EdgeTools
import Foundation
import MLXLMCommon

// MARK: - GenericVLMMLXProfile

public struct GenericVLMMLXProfile: MLXVLMModelProfile {
  public typealias Prompt = EdgeToolsLLMPrompt
  public typealias ToolCallParser = QwenJSONToolCallParser
  public typealias GenerateParameters = DefaultMLXGenerateParameters
  public typealias GrammarCompiler = XGRCompiler
  public typealias GrammarContext = XGRGrammarContext

  public static func toolCallGrammar(
    tools: [EdgeToolDefinition],
    range: GrammarToolCallRange
  ) throws -> XGRGrammar {
    try .qwenJSON(tools: tools, range: range)
  }

  public static nonisolated(nonsending) func input(
    prompt: EdgeToolsLLMPrompt,
    tools: [EdgeToolDefinition],
    tokenizer _: any EdgeToolsTokenizer,
    processor: (any UserInputProcessor)?
  ) async throws -> LMInput {
    guard let processor else {
      throw EdgeToolsError(
        code: .failedToLoadConfiguration,
        message: "Could not load model configuration."
      )
    }
    return try await processor.prepare(input: try prompt.genericVLMUserInput(tools: tools))
  }
}

public typealias GenericVLMMLXModelEngine = MLXEngine<GenericVLMMLXProfile>

// MARK: - Prompt Conversion

extension EdgeToolsLLMPrompt {
  fileprivate func genericVLMUserInput(tools: [EdgeToolDefinition]) throws -> UserInput {
    try self.mlxUserInput(tools: tools) { message in
      switch message {
      case .system:
        return try message.mlxMessage()
      case .user(let text, let messageImages, audio: _):
        var content: [MLXLMCommon.Message] = messageImages.map { _ in ["type": "image"] }
        content.append(["type": "text", "text": text])
        return ["role": "user", "content": content]
      case .assistant, .tool:
        var result = try message.mlxMessage()
        if let text = result["content"] as? String {
          result["content"] = [["type": "text", "text": text]] as [MLXLMCommon.Message]
        }
        return result
      @unknown default:
        return try message.mlxMessage()
      }
    }
  }
}
