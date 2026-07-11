import CustomDump
import Needle
import Testing

@Suite
struct `GrammarBitmask tests` {
  @Test
  func `Empty Bitmask Is Zeroed At Proper Size`() {
    let mask = GrammarBitmask()
    expectNoDifference(mask.storage.count, 256)
    for i in 0..<8192 {
      expectNoDifference(mask[i], false)
    }
  }

  @Test
  func `Count`() {
    let mask = GrammarBitmask()
    expectNoDifference(mask.count, 8192)
  }

  @Test
  func `Set Bool Basics`() {
    var mask = GrammarBitmask()

    mask[0] = true
    expectNoDifference(mask[0], true)
    expectNoDifference(mask.storage[0], 0x0000_0001)

    mask[69] = true
    expectNoDifference(mask[69], true)
    expectNoDifference(mask.storage[2], 32)

    mask[69] = false
    expectNoDifference(mask[69], false)
    expectNoDifference(mask.storage[2], 0)
  }

  @Test
  func `Mutate Through Storage`() {
    var mask = GrammarBitmask()
    mask.storage[0] = 0x0000_0001
    expectNoDifference(mask[0], true)

    mask.storage[2] = 32
    expectNoDifference(mask[69], true)
  }
}
