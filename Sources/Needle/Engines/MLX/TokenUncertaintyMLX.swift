#if SwiftNeedleMLX
  import Foundation
  import MLX

  // TODO: - Make Public?

  func tokenUncertaintyMLX(logits: MLXArray) -> Float {
    let values = logits.flattened().asArray(Float.self)
    var best = -Float.infinity
    var second = -Float.infinity
    for value in values {
      guard value.isFinite && value != -Float.infinity else { continue }
      if value > best {
        second = best
        best = value
      } else if value > second {
        second = value
      }
    }
    let margin = max(-60, min(60, best - second))
    let confidence = 1 / (1 + exp(-margin))
    return max(0, min(1, 1 - confidence))
  }
#endif
