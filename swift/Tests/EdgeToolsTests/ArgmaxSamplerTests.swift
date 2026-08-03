import CustomDump
import Testing

@testable import EdgeTools

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
#endif

@Suite
struct `SIMD argmax tests` {
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

#if ONNXCore
  private func sampleONNXArgmax(values: [Float]) async throws -> EdgeToolsToken.ID {
    let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: values.count)
    _ = buffer.initialize(from: values)
    defer {
      buffer.deinitialize()
      buffer.deallocate()
    }

    let view = ONNXTensorView(
      unsafelyWrapping: buffer.baseAddress,
      shape: [values.count]
    )
    return try await EdgeToolsSampler<ONNXTensorView<Float>>.argmax.sample(logits: view)
  }

  @Suite
  struct `ONNXArgmaxSampler tests` {
    @Test(
      arguments: [
        ([Float](), 0),
        ([3], 0),
        ([1, 4, 2], 1),
        ([4, 4, 2], 0),
        (Array(repeating: -Float.infinity, count: 33), 0)
      ]
    )
    func `Samples Expected Token`(
      values: [Float],
      expectedToken: EdgeToolsToken.ID
    ) async throws {
      let token = try await sampleONNXArgmax(values: values)

      expectNoDifference(token, expectedToken)
    }

    @Test
    func `Samples Tail Token`() async throws {
      var values = Array(repeating: Float(-1), count: 67)
      values[66] = 1

      let token = try await sampleONNXArgmax(values: values)

      expectNoDifference(token, 66)
    }
  }
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  @Suite
  struct `CoreAIArgmaxSampler tests` {
    @Test(
      arguments: [
        ([3], 0),
        ([1, 4, 2], 1),
        ([4, 4, 2], 0),
        (Array(repeating: -Float.infinity, count: 33), 0)
      ]
    )
    @available(anyAppleOS 27.0, *)
    func `Samples Expected Token`(
      values: [Float],
      expectedToken: EdgeToolsToken.ID
    ) async throws {
      let logits = NDArray(scalars: values, shape: [1, values.count])
      let token = try await EdgeToolsSampler<NDArray>.argmax.sample(logits: logits)

      expectNoDifference(token, expectedToken)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Samples Tail Token`() async throws {
      var values = Array(repeating: Float(-1), count: 67)
      values[66] = 1
      let logits = NDArray(scalars: values, shape: [1, values.count])

      let token = try await EdgeToolsSampler<NDArray>.argmax.sample(logits: logits)

      expectNoDifference(token, 66)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Samples Token From Two Hundred Fifty Six Thousand Vocabulary`() async throws {
      var values = Array(repeating: Float(-1), count: 256 * 1024)
      values[values.count - 17] = 1
      let logits = NDArray(scalars: values, shape: [1, values.count])

      let token = try await EdgeToolsSampler<NDArray>.argmax.sample(logits: logits)

      expectNoDifference(token, values.count - 17)
    }
  }
#endif
