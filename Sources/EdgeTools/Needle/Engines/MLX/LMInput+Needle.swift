#if MLX && canImport(MLX)
  import Foundation
  import MLX
  import MLXLMCommon

  extension LMInput {
    public static func needle(
      prompt: EdgeToolsPrompt,
      using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
    ) throws -> Self {
      let tokens = tokenizer.encode(text: prompt.needleFormatted())
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }
#endif
