// MARK: - Core AI

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import Accelerate
  import CoreAI

  @available(anyAppleOS 27.0, *)
  public struct CoreAIArgmaxSampler: EdgeToolsSampler {
    public init() {}

    public func sample(logits: NDArray) -> EdgeToolsToken.ID {
      let view = logits.view(as: Float.self)
      let vocabularySize = view.shape[1]
      guard vocabularySize > 0 else { return 0 }

      if view.isContiguous {
        return view.withUnsafePointer { pointer, _, _ in
          _argmaxVDSP(UnsafeBufferPointer(start: pointer, count: vocabularySize))
        }
      } else {
        return _argmaxScalar(count: vocabularySize) { view[scalarAt: [0, $0]] }
      }
    }
  }

  @inline(always)
  func _argmaxVDSP(_ values: UnsafeBufferPointer<Float>) -> Int {
    guard !values.isEmpty else { return 0 }
    var maximum = Float.zero
    var index = vDSP_Length.zero
    vDSP_maxvi(values.baseAddress!, 1, &maximum, &index, vDSP_Length(values.count))
    return Int(index)
  }


  @inline(always)
  func _argmaxScalar(
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
#endif

// MARK: - Core ML

#if CoreML && canImport(CoreML)
  import CoreML

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public struct CoreMLArgmaxSampler: EdgeToolsSampler {
    public init() {}

    public func sample(logits: MLTensor) async -> EdgeToolsToken.ID {
      let indices = await logits.argmax(alongAxis: 1).shapedArray(of: Int32.self).scalars
      return EdgeToolsToken.ID(indices.first ?? 0)
    }
  }
#endif
