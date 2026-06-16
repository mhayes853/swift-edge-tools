#if SwiftNeedleMLX && SwiftNeedleTokenizers
  import Foundation
  import MLX
  import MLXLMCommon
  import Tokenizers

  extension LMInput {
    public static func needle(
      prompt: NeedlePrefillablePrompt,
      using tokenizer: some TokenizingModel
    ) -> Self {
      let tokenStrings = tokenizer.tokenize(text: prompt.formatted())
      let tokens = tokenStrings.compactMap(tokenizer.convertTokenToId)
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }

    public static func needle(
      prompt: NeedlePrompt,
      using tokenizer: some TokenizingModel
    ) throws -> Self {
      let tokenStrings = try tokenizer.tokenize(text: prompt.formatted())
      let tokens = tokenStrings.compactMap(tokenizer.convertTokenToId)
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }
#endif
