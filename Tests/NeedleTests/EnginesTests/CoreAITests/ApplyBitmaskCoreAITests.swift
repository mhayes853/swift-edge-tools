#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CustomDump
  import Needle
  import Testing
  import CoreAI

  @Suite
  struct `ApplyBitmaskCoreAI tests` {
    @Test
    @available(anyAppleOS 27.0, *)
    func `Filters Masked Tokens`() {

      var mask = NeedleGrammarBitmask(storage: [0, 0, 0])
      for i in 0..<mask.count {
        mask[i] = true
      }
      let expectedIndicies = [0, 6, 8, 15, 31, 32, 63]
      for index in expectedIndicies {
        mask[index] = false
      }

      let initialLogits = NDArray(scalars: (0..<64).map(Float.init), shape: [1, 64])
      let logits = applyBitmaskCoreAI(logits: initialLogits, mask: mask)
      let view = logits.view(as: Float.self)
      let filtered = (0..<64)
        .compactMap { index in
          view[scalarAt: [0, index]] == -.infinity ? index : nil
        }

      expectNoDifference(filtered, expectedIndicies)
    }
  }
#endif
