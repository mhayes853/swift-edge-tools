import CustomDump
import EdgeTools
import Testing

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
  func `Bitmask Supports Byte Aligned Sizes`() {
    let mask = GrammarBitmask(bitCount: 8)

    expectNoDifference(mask.storage.count, 1)
    expectNoDifference(mask.count, 8)
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
