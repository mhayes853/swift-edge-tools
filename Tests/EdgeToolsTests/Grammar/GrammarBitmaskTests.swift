import CustomDump
import EdgeTools
import Testing

#if MLX && canImport(MLX)
  import MLX
#endif
#if CoreML && canImport(CoreML)
  import CoreML
#endif
#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
#endif

@Suite
struct GrammarBitmaskConsolidatedTests {
  @Suite
  struct `GrammarBitmask tests` {
    @Test
    func `Bitmask Is Zeroed At Requested Size`() {
      let mask = GrammarBitmask(bitCount: 8192)

      expectNoDifference(mask.storage.count, 1024)
      expectNoDifference(mask.count, 8192)
      expectNoDifference(mask.allSatisfy { !$0 }, true)
    }

    @Test
    func `Bitmask Can Allow Every Token`() {
      let mask = GrammarBitmask(bitCount: 24, repeating: true)

      expectNoDifference(mask.storage, [.max, .max, .max])
      expectNoDifference(mask.allSatisfy { $0 }, true)
    }

    @Test
    func `Set Bool Basics`() {
      var mask = GrammarBitmask(bitCount: 72)

      mask[0] = true
      expectNoDifference(mask[0], true)
      expectNoDifference(mask.storage[0], 1)

      mask[69] = true
      expectNoDifference(mask[69], true)
      expectNoDifference(mask.storage[8], 32)

      mask[69] = false
      expectNoDifference(mask[69], false)
      expectNoDifference(mask.storage[8], 0)
    }

    @Test
    func `Mutate Through Storage`() {
      var mask = GrammarBitmask(bitCount: 72)
      mask.storage[0] = 1
      expectNoDifference(mask[0], true)

      mask.storage[8] = 32
      expectNoDifference(mask[69], true)
    }
  }

  #if XGrammar
    @Suite
    struct `ApplyBitmask tests` {
      #if MLX && canImport(MLX)
        @Test(.enabledIfXcode())
        func `MLX Filters Masked Tokens`() {
          let mask = Self.mask()
          let initialLogits = MLXArray((0..<64).map(Float.init))
          let logits = applyBitmaskMLX(logits: initialLogits[.newAxis, 0...], mask: mask)
          let filtered = logits[0].enumerated()
            .compactMap { index, logit in
              logit.item(Float.self) == -.infinity ? index : nil
            }

          expectNoDifference(filtered, Self.expectedIndices)
        }
      #endif

      #if CoreML && canImport(CoreML)
        @Test
        @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
        func `Core ML Filters Masked Tokens`() async {
          let mask = Self.mask()
          let initialLogits = MLTensor(shape: [1, 64], scalars: (0..<64).map(Float.init))
          let logits = await applyBitmaskCoreML(logits: initialLogits, mask: mask)
            .shapedArray(of: Float.self)
          let filtered = (0..<64)
            .filter { logits[scalarAt: [0, $0]] == -.infinity }

          expectNoDifference(filtered, Self.expectedIndices)
        }
      #endif

      #if swift(>=6.4) && CoreAI && canImport(CoreAI)
        @Test
        @available(anyAppleOS 27.0, *)
        func `Core AI Filters Masked Tokens`() {
          let mask = Self.mask()
          var initialLogits = NDArray(scalars: (0..<64).map(Float.init), shape: [1, 64])
          let logits = applyBitmaskCoreAI(logits: &initialLogits, mask: mask)
          let view = logits.view(as: Float.self)
          let filtered = (0..<64)
            .compactMap { index in
              view[scalarAt: [0, index]] == -.infinity ? index : nil
            }

          expectNoDifference(filtered, Self.expectedIndices)
        }
      #endif

      private static let expectedIndices = [0, 6, 8, 15, 31, 32, 63]

      private static func mask() -> GrammarBitmask {
        var mask = GrammarBitmask(bitCount: 64, repeating: true)
        for index in Self.expectedIndices {
          mask[index] = false
        }
        return mask
      }
    }
  #endif
}
