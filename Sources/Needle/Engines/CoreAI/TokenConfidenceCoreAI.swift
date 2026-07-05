#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import Foundation

  // MARK: - NeedleConfidenceState (CoreAI)

  @available(anyAppleOS 27.0, *)
  extension NeedleConfidenceState {
    mutating func add(logits: NDArray) throws {
      self.add(confidence: try tokenConfidenceCoreAI(logits: logits))
    }
  }

  // MARK: - Uncertainty

  @available(anyAppleOS 27.0, *)
  private func tokenConfidenceCoreAI(logits: NDArray) throws -> Float {
    let view = logits.view(as: Float.self)
    let vocabularySize = view.shape[1]

    var top1 = -Float.infinity
    var top2 = -Float.infinity
    for j in 0..<vocabularySize {
      let value = view[scalarAt: [0, j]]
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
