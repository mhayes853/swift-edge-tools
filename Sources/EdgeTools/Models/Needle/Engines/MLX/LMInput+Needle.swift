#if MLX && canImport(MLX)
  import Foundation
  import MLX
  import MLXLMCommon

  extension LMInput {
    public static func needle(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
    ) throws -> Self {
      let tokens = tokenizer.encode(text: try prompt.formatted(tools: tools))
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }
#endif
