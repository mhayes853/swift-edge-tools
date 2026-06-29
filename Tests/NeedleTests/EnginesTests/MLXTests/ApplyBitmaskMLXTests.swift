#if MLX
  import CustomDump
  import Needle
  import Testing
  import MLX

  @Suite(.enabledIfXcode())
  struct `ApplyBitmaskMLX tests` {
    @Test
    func `Filters Masked Tokens`() {
      var mask = NeedleGrammarBitmask(storage: [0, 0])
      for i in 0..<mask.count {
        mask[i] = true
      }
      mask[0] = false
      mask[6] = false

      let initialLogits = MLXArray(
        [4.32, 7.98, -1.28, 7.267, 8.1, -92.32, 0.02, 8.29] as [Float],
      )
      let logits = applyBitmaskMLX(logits: initialLogits[.newAxis, 0...], mask: mask)
      let filtered = logits[0].enumerated()
        .compactMap { (i, logit) in
          logit.item(Float.self) == -.infinity ? i : nil
        }
      expectNoDifference(filtered, [0, 6])
    }
  }
#endif
