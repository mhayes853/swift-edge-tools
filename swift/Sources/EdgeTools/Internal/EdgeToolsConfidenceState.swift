#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Musl)
  import Musl
#elseif canImport(WASILibc)
  import WASILibc
#endif

#if MLX && canImport(MLX)
  import MLX
#endif

#if CoreML && canImport(CoreML)
  import CoreML
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
#endif

// MARK: - EdgeToolsConfidenceState

struct EdgeToolsConfidenceState {
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

// MARK: - ONNX

#if ONNXCore
  func tokenConfidenceONNX(logits: [Float]) -> Float {
    var top1 = -Float.infinity
    var top2 = -Float.infinity
    for value in logits {
      if value > top1 {
        top2 = top1
        top1 = value
      } else if value > top2 {
        top2 = value
      }
    }
    let margin = Swift.min(Swift.max(top1 - top2, -60), 60)
    return 1 / (1 + exp(-margin))
  }
#endif

// MARK: - MLX

#if MLX && canImport(MLX)
  func tokenConfidenceMLX(logits: MLXArray) -> Float {
    let top2 = top(logits.flattened(), k: 2)
    let margin = clip(top2[1] - top2[0], min: -60.0, max: 60.0)
    return (1.0 / (1.0 + exp(-margin))).item(Float.self)
  }
#endif

// MARK: - Core ML

#if CoreML && canImport(CoreML)
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  func tokenConfidenceCoreML(logits: MLTensor) async -> Float {
    let values = Array((await logits.cast(to: Float.self).shapedArray(of: Float.self)).scalars)
      .sorted(by: >)
    let top1 = values.first ?? -.infinity
    let top2 = values.dropFirst().first ?? -.infinity
    let margin = Swift.min(Swift.max(top1 - top2, -60.0), 60.0)
    return 1.0 / (1.0 + exp(-margin))
  }
#endif

// MARK: - Core AI

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  @available(anyAppleOS 27.0, *)
  func tokenConfidenceCoreAI(logits: NDArray) throws -> Float {
    let view = logits.view(as: Float.self)
    let vocabularySize = view.shape[1]

    var top1 = -Float.infinity
    var top2 = -Float.infinity
    for tokenIndex in 0..<vocabularySize {
      let value = view[scalarAt: [0, tokenIndex]]
      if value > top1 {
        top2 = top1
        top1 = value
      } else if value > top2 {
        top2 = value
      }
    }

    let margin = Swift.min(Swift.max(top1 - top2, -60.0), 60.0)
    return 1.0 / (1.0 + exp(-margin))
  }
#endif
