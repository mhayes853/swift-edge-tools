#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Musl)
  import Musl
#elseif canImport(WASILibc)
  import WASILibc
#elseif canImport(Android)
  import Android
#elseif canImport(ucrt)
  import ucrt
#endif

#if canImport(simd)
  import simd
#endif

// MARK: - ConfidenceState

struct ConfidenceState {
  private(set) var perTokenConfidences = [Float]()
  private var totalSum = Float(0)

  var mean: Float? {
    !self.perTokenConfidences.isEmpty
      ? self.totalSum / Float(self.perTokenConfidences.count)
      : nil
  }

  mutating func add(confidence: Float) {
    self.perTokenConfidences.append(confidence)
    self.totalSum += confidence
  }
}

// MARK: - Top 2

@inline(always)
package func top2SIMD(_ values: UnsafeBufferPointer<Float>) -> (top1: Float, top2: Float) {
  let width = SIMD16<Float>.scalarCount
  var laneTop1 = SIMD16<Float>(repeating: -.infinity)
  var laneTop2 = SIMD16<Float>(repeating: -.infinity)
  var index = 0
  while index + width <= values.count {
    let pointer = UnsafeRawPointer(values.baseAddress!.advanced(by: index))
    let candidates = pointer.loadUnaligned(as: SIMD16<Float>.self)
    laneTop2 = top2SIMDMax(laneTop2, top2SIMDMin(laneTop1, candidates))
    laneTop1 = top2SIMDMax(laneTop1, candidates)
    index += width
  }

  var top1 = -Float.infinity
  var top2 = -Float.infinity
  for lane in 0..<width {
    updateTop2(top1: &top1, top2: &top2, with: laneTop1[lane])
    updateTop2(top1: &top1, top2: &top2, with: laneTop2[lane])
  }
  return top2Scalar(count: values.count, from: index, top1: top1, top2: top2) { values[$0] }
}

@inline(always)
package func top2SIMDMax(_ lhs: SIMD16<Float>, _ rhs: SIMD16<Float>) -> SIMD16<Float> {
  #if canImport(simd)
    simd_max(lhs, rhs)
  #else
    lhs.replacing(with: rhs, where: rhs .> lhs)
  #endif
}

@inline(always)
package func top2SIMDMin(_ lhs: SIMD16<Float>, _ rhs: SIMD16<Float>) -> SIMD16<Float> {
  #if canImport(simd)
    simd_min(lhs, rhs)
  #else
    lhs.replacing(with: rhs, where: rhs .< lhs)
  #endif
}

@inline(always)
package func top2Scalar(
  count: Int,
  from initialIndex: Int = 0,
  top1 initialTop1: Float = -.infinity,
  top2 initialTop2: Float = -.infinity,
  valueAt: (Int) -> Float
) -> (top1: Float, top2: Float) {
  var top1 = initialTop1
  var top2 = initialTop2
  for index in initialIndex..<count {
    updateTop2(top1: &top1, top2: &top2, with: valueAt(index))
  }
  return (top1, top2)
}

@inline(always)
package func updateTop2(top1: inout Float, top2: inout Float, with value: Float) {
  if value > top1 {
    top2 = top1
    top1 = value
  } else if value > top2 {
    top2 = value
  }
}

@inline(always)
func tokenConfidence(top1: Float, top2: Float) -> Float {
  let margin = top1 - top2
  guard !margin.isNaN else { return 0 }
  return Float(1 / (1 + exp(Double(-Swift.min(Swift.max(margin, -60), 60)))))
}

@inline(always)
func tokenConfidence(unorderedPair values: some Collection<Float>) -> Float {
  guard let first = values.first else { return 0 }
  let second = values.dropFirst().first ?? -.infinity
  return tokenConfidence(top1: Swift.max(first, second), top2: Swift.min(first, second))
}

// MARK: - ONNX

#if ONNXCore
  func tokenConfidenceONNX(logits: Span<Float>) -> Float {
    let top = logits.withUnsafeBufferPointer { top2SIMD($0) }
    return tokenConfidence(top1: top.top1, top2: top.top2)
  }
#endif
