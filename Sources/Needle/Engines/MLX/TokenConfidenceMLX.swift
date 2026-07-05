#if MLX && canImport(MLX)
  import Foundation
  import MLX

  // MARK: - NeedleConfidenceState (MLX)

  extension NeedleConfidenceState {
    mutating func add(logits: MLXArray) {
      self.add(confidence: tokenConfidenceMLX(logits: logits))
    }
  }

  // MARK: - Uncertainty

  private func tokenConfidenceMLX(logits: MLXArray) -> Float {
    let top2 = top(logits.flattened(), k: 2)
    let margin = clip(top2[1] - top2[0], min: -60.0, max: 60.0)
    return (1.0 / (1.0 + exp(-margin))).item(Float.self)
  }
#endif