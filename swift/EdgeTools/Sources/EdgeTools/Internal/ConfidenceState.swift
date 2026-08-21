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

// MARK: - ConfidenceState

struct ConfidenceState {
  private(set) var perTokenConfidences = [Float]()
  private var totalSum = Float(0)
  private var count = 0

  var mean: Float? {
    self.count > 0
      ? self.totalSum / Float(self.count)
      : nil
  }

  mutating func add(confidence: Float, options: EdgeToolsConfidenceOptions) {
    if options.contains(.generation) {
      self.totalSum += confidence
      self.count += 1
    }
    if options.contains(.perToken) {
      self.perTokenConfidences.append(confidence)
    }
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
