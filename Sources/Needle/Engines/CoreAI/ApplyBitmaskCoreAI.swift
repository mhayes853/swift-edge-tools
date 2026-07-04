#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI

  @available(anyAppleOS 27.0, *)
  public func applyBitmaskCoreAI(logits: NDArray, mask: NeedleGrammarBitmask) -> NDArray {
    fatalError()
  }
#endif
