import CustomDump
import EdgeToolsCLI
import Testing

@Suite
struct `MemoryByteCount tests` {
  @Test
  func `Peak Resident Memory Is Expressed In Bytes`() {
    expectNoDifference(MemoryByteCount.peakResident.bytes > 1_048_576, true)
  }
}
