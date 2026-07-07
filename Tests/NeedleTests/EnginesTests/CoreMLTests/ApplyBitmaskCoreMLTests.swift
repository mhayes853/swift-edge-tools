#if CoreML && canImport(CoreML)
  import CoreML
  import Needle
  import Testing

  @Suite
  struct `ApplyBitmaskCoreML tests` {
    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Filters Masked Tokens`() async {
      var mask = NeedleGrammarBitmask(storage: [0, 0, 0])
      for index in 0..<mask.count {
        mask[index] = true
      }
      let expectedIndices = [0, 6, 8, 15, 31, 32, 63]
      for index in expectedIndices {
        mask[index] = false
      }

      let initialLogits = MLTensor(shape: [1, 64], scalars: (0..<64).map(Float.init))
      let logits = await applyBitmaskCoreML(logits: initialLogits, mask: mask)
        .shapedArray(of: Float.self)
      let filtered = (0..<64)
        .filter { logits[scalarAt: [0, $0]] == -.infinity }
      #expect(filtered == expectedIndices)
    }
  }
#endif
