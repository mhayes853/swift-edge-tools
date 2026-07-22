#if canImport(Accelerate)
  import Accelerate
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
#endif

#if CoreML && canImport(CoreML)
  import CoreML
#endif

// MARK: - EdgeToolsSampler

public protocol EdgeToolsSampler<Logits> {
  associatedtype Logits

  func sample(logits: Logits) async throws -> EdgeToolsToken.ID
}

// MARK: - Argmax

@inline(always)
func argmaxContiguous(_ values: UnsafeBufferPointer<Float>) -> Int {
  #if canImport(Accelerate)
    argmaxVDSP(values)
  #else
    argmaxSIMD(values)
  #endif
}

#if canImport(Accelerate)
  @inline(always)
  func argmaxVDSP(_ values: UnsafeBufferPointer<Float>) -> Int {
    guard !values.isEmpty else { return 0 }
    var maximum = Float.zero
    var index = vDSP_Length.zero
    vDSP_maxvi(values.baseAddress!, 1, &maximum, &index, vDSP_Length(values.count))
    return Int(index)
  }
#endif

@inline(always)
func argmaxSIMD(_ values: UnsafeBufferPointer<Float>) -> Int {
  guard !values.isEmpty else { return 0 }
  let width = SIMD8<Float>.scalarCount
  var bestValues = SIMD8<Float>(repeating: -.infinity)
  var bestIndices = SIMD8<Int32>(repeating: 0)
  var index = 0
  while index + width <= values.count {
    updateArgmaxSIMD(
      values,
      at: index,
      bestValues: &bestValues,
      bestIndices: &bestIndices
    )
    index += width
  }

  var bestValue = -Float.infinity
  var bestIndex = 0
  reduceArgmaxSIMD(
    values: bestValues,
    indices: bestIndices,
    bestValue: &bestValue,
    bestIndex: &bestIndex
  )
  return argmaxScalar(
    count: values.count,
    from: index,
    bestValue: bestValue,
    bestIndex: bestIndex,
    valueAt: { values[$0] }
  )
}

@inline(always)
private func updateArgmaxSIMD(
  _ values: UnsafeBufferPointer<Float>,
  at index: Int,
  bestValues: inout SIMD8<Float>,
  bestIndices: inout SIMD8<Int32>
) {
  let pointer = UnsafeRawPointer(values.baseAddress!.advanced(by: index))
  let candidates = pointer.loadUnaligned(as: SIMD8<Float>.self)
  let replacementMask = candidates .> bestValues
  let firstIndex = Int32(index)
  let candidateIndices = SIMD8<Int32>(
    firstIndex,
    firstIndex + 1,
    firstIndex + 2,
    firstIndex + 3,
    firstIndex + 4,
    firstIndex + 5,
    firstIndex + 6,
    firstIndex + 7
  )
  bestValues.replace(with: candidates, where: replacementMask)
  bestIndices.replace(with: candidateIndices, where: replacementMask)
}

@inline(always)
private func reduceArgmaxSIMD(
  values: SIMD8<Float>,
  indices: SIMD8<Int32>,
  bestValue: inout Float,
  bestIndex: inout Int
) {
  for lane in 0..<SIMD8<Float>.scalarCount {
    let value = values[lane]
    let index = Int(indices[lane])
    if value > bestValue || (value == bestValue && index < bestIndex) {
      bestValue = value
      bestIndex = index
    }
  }
}

@inline(always)
func argmaxScalar(
  count: Int,
  from initialIndex: Int = 0,
  bestValue initialBestValue: Float = -.infinity,
  bestIndex initialBestIndex: Int = 0,
  valueAt: (Int) -> Float
) -> Int {
  var bestIndex = initialBestIndex
  var bestValue = initialBestValue
  for index in initialIndex..<count {
    let value = valueAt(index)
    if value > bestValue {
      bestIndex = index
      bestValue = value
    }
  }
  return bestIndex
}

// MARK: - ONNX

#if ONNXCore
  public struct ONNXArgmaxSampler: EdgeToolsSampler, Hashable, Sendable {
    public init() {}

    public func sample(logits: [Float]) -> EdgeToolsToken.ID {
      logits.withUnsafeBufferPointer { argmaxContiguous($0) }
    }
  }
#endif

// MARK: - Core AI

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  @available(anyAppleOS 27.0, *)
  public struct CoreAIArgmaxSampler: EdgeToolsSampler {
    public init() {}

    public func sample(logits: NDArray) -> EdgeToolsToken.ID {
      let view = logits.view(as: Float.self)
      let vocabularySize = view.shape[1]
      guard vocabularySize > 0 else { return 0 }

      if view.isContiguous {
        return view.withUnsafePointer { pointer, _, _ in
          argmaxContiguous(UnsafeBufferPointer(start: pointer, count: vocabularySize))
        }
      } else {
        return argmaxScalar(count: vocabularySize) { view[scalarAt: [0, $0]] }
      }
    }
  }
#endif

// MARK: - Core ML

#if CoreML && canImport(CoreML)
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public struct CoreMLArgmaxSampler: EdgeToolsSampler {
    public init() {}

    public func sample(logits: MLTensor) async -> EdgeToolsToken.ID {
      let indices = await logits.argmax(alongAxis: 1).shapedArray(of: Int32.self).scalars
      return EdgeToolsToken.ID(indices.first ?? 0)
    }
  }
#endif
