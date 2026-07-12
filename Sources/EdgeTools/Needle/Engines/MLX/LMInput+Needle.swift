#if MLX && canImport(MLX)
  import Foundation
  import MLX
  import MLXLMCommon
  import Tokenizers

  extension LMInput {
    public static func needle(
      prompt: EdgeToolsPrompt,
      using tokenizer: some Tokenizers.Tokenizer
    ) throws -> Self {
      let tokens = tokenizer.encode(text: prompt.needleFormatted(), addSpecialTokens: false)
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }
#endif
