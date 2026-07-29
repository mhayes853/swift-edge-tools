import CustomDump
import Testing
import EdgeTools

@Suite
struct `Top2SIMD tests` {
  @Test(arguments: top2Cases())
  func `Finds Expected Top Two`(values: [Float], expected: (Float, Float)) {
    let top = values.withUnsafeBufferPointer { top2SIMD($0) }

    expectNoDifference(top.top1, expected.0)
    expectNoDifference(top.top2, expected.1)
  }

  @Test
  func `Finds Top Two Split Across Vector Lanes And Remainder`() {
    var values = Array(repeating: Float(-1), count: 67)
    values[3] = 10
    values[66] = 5

    let top = values.withUnsafeBufferPointer { top2SIMD($0) }

    expectNoDifference(top.top1, 10)
    expectNoDifference(top.top2, 5)
  }

  @Test
  func `Finds Duplicated Maximum As Both Top Values`() {
    var values = Array(repeating: Float(-1), count: 64)
    values[2] = 7
    values[40] = 7

    let top = values.withUnsafeBufferPointer { top2SIMD($0) }

    expectNoDifference(top.top1, 7)
    expectNoDifference(top.top2, 7)
  }

  @Test
  func `Finds Duplicated Maximum Within A Single Lane`() {
    var values = Array(repeating: Float(-1), count: 64)
    values[5] = 7
    values[13] = 7

    let top = values.withUnsafeBufferPointer { top2SIMD($0) }

    expectNoDifference(top.top1, 7)
    expectNoDifference(top.top2, 7)
  }
}

private func top2Cases() -> [([Float], (Float, Float))] {
  [
    ([Float](), (-.infinity, -.infinity)),
    ([3], (3, -.infinity)),
    ([1, 4, 2], (4, 2)),
    ([4, 4, 2], (4, 4)),
    ([2, 4], (4, 2)),
    (Array(repeating: -Float.infinity, count: 33), (-.infinity, -.infinity)),
    ((0..<64).map(Float.init), (63, 62)),
    ((0..<64).map { Float(63 - $0) }, (63, 62))
  ]
}

