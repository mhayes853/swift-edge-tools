#if MLX && canImport(MLX)
  import Foundation
  import MLX
  import MLXLMCommon

  extension LMInput {
    public static func needle(
      prompt: NeedlePrompt,
      using tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
    ) throws -> Self {
      let tokens = tokenizer.encode(text: try prompt.formatted())
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }
#endif
