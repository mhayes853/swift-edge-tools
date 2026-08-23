#if MLX && canImport(MLX)
  import EdgeToolsCore
  import MLX
  import MLXLMCommon

  // MARK: - MLXPrefixCache

  struct MLXPrefixState {
    let input: LMInput
    var tokenIds: [EdgeToolsToken.ID]
    var cache: [any KVCache]
    var output: LMOutput
    let context: EdgeToolsLLMPrefillContext
  }

  struct MLXPrefixCache {
    let state: MLXPrefixState
    let inputKind: EdgeToolsLLMInputKind

    func preparedInput(
      for context: EdgeToolsLLMPrefillContext,
      kind: EdgeToolsLLMInputKind
    ) -> LMInput? {
      guard self.state.context == context, self.inputKind == kind else { return nil }
      return self.state.input
    }

    func state(
      continuingWith tokenIds: [EdgeToolsToken.ID],
      input: LMInput,
      context: EdgeToolsLLMPrefillContext
    ) -> MLXPrefixState? {
      guard tokenIds.starts(with: self.state.tokenIds),
        mlxPrefillContextMatches(
          cachedInput: self.state.input,
          input: input,
          cachedContext: self.state.context,
          inputContext: context
        )
      else {
        return nil
      }
      var state = self.state
      state.cache = self.state.cache.map { $0.copy() }
      return state
    }
  }

  // MARK: - Prefix Matching

  private func mlxPrefillContextMatches(
    cachedInput: LMInput,
    input: LMInput,
    cachedContext: EdgeToolsLLMPrefillContext,
    inputContext: EdgeToolsLLMPrefillContext
  ) -> Bool {
    mlxTextMasksHaveSamePrefix(cachedInput.text.mask, input.text.mask)
      && cachedContext.continuation(in: inputContext) == .textOnly
  }

  private func mlxTextMasksHaveSamePrefix(_ lhs: MLXArray?, _ rhs: MLXArray?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      return true
    case (.some(let lhs), .some(let rhs)):
      guard lhs.ndim == rhs.ndim, lhs.shape.dropLast() == rhs.shape.dropLast(),
        lhs.dim(-1) <= rhs.dim(-1)
      else {
        return false
      }
      return lhs.dtype == rhs.dtype
        && lhs.shape.dropLast() == rhs.shape.dropLast()
        && lhs.arrayEqual(rhs[.ellipsis, ..<lhs.dim(-1)]).item(Bool.self)
    default:
      return false
    }
  }

  // MARK: - Text Slices

  func mlxTextSuffix(_ text: LMInput.Text, from index: Int) -> LMInput.Text {
    let tokens =
      if text.tokens.ndim == 1 {
        text.tokens[index...][.newAxis]
      } else {
        text.tokens[.ellipsis, index...]
      }
    let mask = text.mask.map { mask in
      if mask.ndim == 1 {
        mask[index...][.newAxis]
      } else {
        mask[.ellipsis, index...]
      }
    }
    return LMInput.Text(tokens: tokens, mask: mask)
  }

  func mlxTextPrefix(_ text: LMInput.Text, count: Int) -> LMInput.Text {
    let tokens =
      if text.tokens.ndim == 1 {
        text.tokens[..<count][.newAxis]
      } else {
        text.tokens[.ellipsis, ..<count]
      }
    let mask = text.mask.map { mask in
      if mask.ndim == 1 {
        mask[..<count][.newAxis]
      } else {
        mask[.ellipsis, ..<count]
      }
    }
    return LMInput.Text(tokens: tokens, mask: mask)
  }
#endif
