#if MLX && canImport(MLX)
  import CustomDump
  import EdgeTools
  import Testing
  import MLX

  @Suite(.enabledIfXcode())
  struct `ApplyBitmaskMLX tests` {
    @Test
    func `Filters Masked Tokens`() {
      var mask = GrammarBitmask(bitCount: 64, repeating: true)
      let expectedIndicies = [0, 6, 8, 15, 31, 32, 63]
      for index in expectedIndicies {
        mask[index] = false
      }

      let initialLogits = MLXArray(
        (0..<64).map(Float.init),
      )
      let logits = applyBitmaskMLX(logits: initialLogits[.newAxis, 0...], mask: mask)
      let filtered = logits[0].enumerated()
        .compactMap { (index, logit) in
          logit.item(Float.self) == -.infinity ? index : nil
        }
      expectNoDifference(filtered, expectedIndicies)
    }
  }
#endif
