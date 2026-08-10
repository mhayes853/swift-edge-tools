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

public struct EdgeToolsSampler<Logits: ~Copyable & ~Escapable> {
  private let body: nonisolated(nonsending) (
    borrowing Logits
  ) async throws -> EdgeToolsToken.ID

  public init(
    _ body: nonisolated(nonsending) @escaping (
      borrowing Logits
    ) async throws -> EdgeToolsToken.ID
  ) {
    self.body = body
  }

  public nonisolated(nonsending) func sample(
    logits: borrowing Logits
  ) async throws -> EdgeToolsToken.ID {
    try await self.body(logits)
  }
}

// MARK: - EdgeToolsFusedSamplingParameters

public struct EdgeToolsFusedSamplingParameters: Hashable, Sendable {
  public var temperature: Float
  public var topK: Int?
  public var topP: Float?
  public var minP: Float?
  public var repetitionPenalty: Float
  public var presencePenalty: Float
  public var repetitionContextSize: Int
  public var seed: UInt64?

  public init(
    temperature: Float = 0.6,
    topK: Int? = nil,
    topP: Float? = nil,
    minP: Float? = nil,
    repetitionPenalty: Float = 1,
    presencePenalty: Float = 0,
    repetitionContextSize: Int = 20,
    seed: UInt64? = nil
  ) {
    precondition(repetitionPenalty > 0, "Repetition penalty must be greater than zero.")
    self.temperature = temperature
    self.topK = topK
    self.topP = topP
    self.minP = minP
    self.repetitionPenalty = repetitionPenalty
    self.presencePenalty = presencePenalty
    self.repetitionContextSize = repetitionContextSize
    self.seed = seed
  }

  public var isGreedy: Bool {
    self.temperature == 0
  }

  public var penalizesHistory: Bool {
    self.repetitionPenalty != 1 || self.presencePenalty != 0
  }
}

// MARK: - EdgeToolsFusedSamplingOverrides

public struct EdgeToolsFusedSamplingOverrides: Hashable, Sendable {
  public var temperature: Float?
  public var topK: Int?
  public var topP: Float?
  public var minP: Float?
  public var repetitionPenalty: Float?
  public var presencePenalty: Float?
  public var repetitionContextSize: Int?
  public var seed: UInt64?

  public init(
    temperature: Float? = nil,
    topK: Int? = nil,
    topP: Float? = nil,
    minP: Float? = nil,
    repetitionPenalty: Float? = nil,
    presencePenalty: Float? = nil,
    repetitionContextSize: Int? = nil,
    seed: UInt64? = nil
  ) {
    self.temperature = temperature
    self.topK = topK
    self.topP = topP
    self.minP = minP
    self.repetitionPenalty = repetitionPenalty
    self.presencePenalty = presencePenalty
    self.repetitionContextSize = repetitionContextSize
    self.seed = seed
  }

  public var isEmpty: Bool {
    self.temperature == nil
      && self.topK == nil
      && self.topP == nil
      && self.minP == nil
      && self.repetitionPenalty == nil
      && self.presencePenalty == nil
      && self.repetitionContextSize == nil
      && self.seed == nil
  }

  public func applying(
    to parameters: EdgeToolsFusedSamplingParameters
  ) -> EdgeToolsFusedSamplingParameters {
    var parameters = parameters
    parameters.temperature = self.temperature ?? parameters.temperature
    parameters.topK = self.topK ?? parameters.topK
    parameters.topP = self.topP ?? parameters.topP
    parameters.minP = self.minP ?? parameters.minP
    parameters.repetitionPenalty = self.repetitionPenalty ?? parameters.repetitionPenalty
    parameters.presencePenalty = self.presencePenalty ?? parameters.presencePenalty
    parameters.repetitionContextSize =
      self.repetitionContextSize ?? parameters.repetitionContextSize
    parameters.seed = self.seed ?? parameters.seed
    return parameters
  }
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
  let width = SIMD16<Float>.scalarCount
  var bestValues = SIMD16<Float>(repeating: -.infinity)
  var bestIndices = SIMD16<Int32>(repeating: 0)
  var index = 0
  while index + width <= values.count {
    updateArgmaxSIMD(values, at: index, bestValues: &bestValues, bestIndices: &bestIndices)
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

private let argmaxLaneOffsets = SIMD16<Int32>(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

@inline(always)
private func updateArgmaxSIMD(
  _ values: UnsafeBufferPointer<Float>,
  at index: Int,
  bestValues: inout SIMD16<Float>,
  bestIndices: inout SIMD16<Int32>
) {
  let pointer = UnsafeRawPointer(values.baseAddress!.advanced(by: index))
  let candidates = pointer.loadUnaligned(as: SIMD16<Float>.self)
  let replacementMask = candidates .> bestValues
  bestValues.replace(with: candidates, where: replacementMask)
  bestIndices.replace(with: argmaxLaneOffsets &+ Int32(index), where: replacementMask)
}

@inline(always)
private func reduceArgmaxSIMD(
  values: SIMD16<Float>,
  indices: SIMD16<Int32>,
  bestValue: inout Float,
  bestIndex: inout Int
) {
  for lane in 0..<SIMD16<Float>.scalarCount {
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
  extension EdgeToolsSampler where Logits == ONNXTensorView<Float> {
    public static var argmax: Self {
      EdgeToolsSampler { logits in
        logits.span.withUnsafeBufferPointer { argmaxContiguous($0) }
      }
    }
  }
#endif

// MARK: - Core AI

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  @available(anyAppleOS 27.0, *)
  extension EdgeToolsSampler where Logits == NDArray {
    public static var argmax: Self {
      EdgeToolsSampler { logits in
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
  }
#endif

// MARK: - Core ML

#if CoreML && canImport(CoreML)
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension EdgeToolsSampler where Logits == MLTensor {
    public static var argmax: Self {
      EdgeToolsSampler { logits in
        let indices = await logits.argmax(alongAxis: 1).shapedArray(of: Int32.self).scalars
        return EdgeToolsToken.ID(indices.first ?? 0)
      }
    }
  }
#endif
