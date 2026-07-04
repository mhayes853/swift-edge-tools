#if MLX && canImport(MLX)
  import CustomDump
  import Needle
  import Testing
  import MLX

  @Suite(.enabledIfXcode())
  struct `ApplyBitmaskMLX tests` {
    @Test
    func `Filters Masked Tokens`() {
      var mask = NeedleGrammarBitmask(storage: [0, 0, 0])
      for i in 0..<mask.count {
        mask[i] = true
      }
      let expectedIndicies = [0, 6, 8, 15, 31, 32, 63]
      for index in expectedIndicies {
        mask[index] = false
      }

      let initialLogits = MLXArray(
        (0..<64).map(Float.init),
      )
      let logits = applyBitmaskMLX(logits: initialLogits[.newAxis, 0...], mask: mask)
      let filtered = logits[0].enumerated()
        .compactMap { (i, logit) in
          logit.item(Float.self) == -.infinity ? i : nil
        }
      expectNoDifference(filtered, expectedIndicies)
    }
  }
#endif
