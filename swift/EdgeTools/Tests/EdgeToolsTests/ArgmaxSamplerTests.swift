import CustomDump
import Testing

@testable import EdgeTools

@Suite
struct `SIMDArgmax tests` {
  @Test(arguments: argmaxSIMDCases())
  func `Samples Expected Index`(values: [Float], expectedIndex: Int) {
    let index = values.withUnsafeBufferPointer { argmaxSIMD($0) }

    expectNoDifference(index, expectedIndex)
  }

  @Test
  func `Samples From Two Hundred Fifty Six Thousand Values`() {
    var values = Array(repeating: Float(-1), count: 256 * 1024)
    values[values.count - 17] = 1

    let index = values.withUnsafeBufferPointer { argmaxSIMD($0) }

    expectNoDifference(index, values.count - 17)
  }
}

private func argmaxSIMDCases() -> [([Float], Int)] {
  let basicCases: [([Float], Int)] = [
    ([Float](), 0),
    ([3], 0),
    ([1, 4, 2], 1),
    ([4, 4, 2], 0),
    (Array(repeating: -Float.infinity, count: 33), 0)
  ]
  let boundaryCases = [0, 7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48, 55, 56, 63]
    .map { index in (argmaxValues(count: 64, maximumAt: index), index) }
  return basicCases + boundaryCases + [(argmaxValues(count: 67, maximumAt: 66), 66)]
}

private func argmaxValues(count: Int, maximumAt index: Int) -> [Float] {
  (0..<count).map { $0 == index ? 1 : -1 }
}
