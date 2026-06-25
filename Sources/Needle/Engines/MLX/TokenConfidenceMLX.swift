#if SwiftNeedleMLX
  import Foundation
  import MLX

  // MARK: - ConfidenceState

  struct NeedleMLXConfidenceState {
    private(set) var tokenUncertainties = [Float]()
    private var totalSum: Float = 0

    var mean: Float? {
      !self.tokenUncertainties.isEmpty
        ? 1 - (self.totalSum / Float(self.tokenUncertainties.count))
        : nil
    }

    mutating func add(logits: MLXArray) {
      let uncertainty = tokenUncertaintyMLX(logits: logits)
      self.tokenUncertainties.append(uncertainty)
      self.totalSum += uncertainty
    }
  }

  // MARK: - Uncertainty

  private func tokenUncertaintyMLX(logits: MLXArray) -> Float {
    let top2 = top(logits.flattened(), k: 2)
    let margin = clip(top2[1] - top2[0], min: -60.0, max: 60.0)
    let confidence = 1.0 / (1.0 + exp(-margin))
    return clip(1.0 - confidence, min: 0.0, max: 1.0).item(Float.self)
  }
#endif
