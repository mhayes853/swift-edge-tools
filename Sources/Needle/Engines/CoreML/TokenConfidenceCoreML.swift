#if CoreML && canImport(CoreML)
  import CoreML
  import Foundation

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleConfidenceState {
    mutating func add(logits: MLTensor) async {
      self.add(confidence: await tokenConfidenceCoreML(logits: logits))
    }
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  private func tokenConfidenceCoreML(logits: MLTensor) async -> Float {
    let values = Array((await logits.cast(to: Float.self).shapedArray(of: Float.self)).scalars)
      .sorted(by: >)
    let top1 = values.first ?? -.infinity
    let top2 = values.dropFirst().first ?? -.infinity
    let margin = Swift.min(Swift.max(top1 - top2, -60.0), 60.0)
    return 1.0 / (1.0 + exp(-margin))
  }
#endif
