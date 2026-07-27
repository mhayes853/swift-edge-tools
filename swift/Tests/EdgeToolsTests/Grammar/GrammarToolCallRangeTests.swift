import CustomDump
import EdgeTools
import Testing

@Suite
struct `GrammarToolCallRange tests` {
  @Test
  func `Empty Half Open Ranges Become Exact Bounds`() {
    expectNoDifference(GrammarToolCallRange.bounded(..<0), .exact(0))
    expectNoDifference(GrammarToolCallRange.bounded(..<(-1)), .exact(-1))
    expectNoDifference(GrammarToolCallRange.bounded(3..<3), .exact(3))
  }
}
