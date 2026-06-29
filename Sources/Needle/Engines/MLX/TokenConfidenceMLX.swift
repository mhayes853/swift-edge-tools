#if MLX && canImport(MLX)
  import Foundation
  import MLX

  // MARK: - ConfidenceState

  struct NeedleMLXConfidenceState {
    private(set) var perTokenConfidences = [Float]()
    private var totalSum: Float = 0

    var mean: Float? {
      !self.perTokenConfidences.isEmpty
        ? self.totalSum / Float(self.perTokenConfidences.count)
        : nil
    }

    mutating func add(logits: MLXArray) {
      let confidence = tokenConfidenceMLX(logits: logits)
      self.perTokenConfidences.append(confidence)
      self.totalSum += confidence
    }
  }

  // MARK: - Uncertainty

  private func tokenConfidenceMLX(logits: MLXArray) -> Float {
    let top2 = top(logits.flattened(), k: 2)
    let margin = clip(top2[1] - top2[0], min: -60.0, max: 60.0)
    return (1.0 / (1.0 + exp(-margin))).item(Float.self)
  }
#endif
