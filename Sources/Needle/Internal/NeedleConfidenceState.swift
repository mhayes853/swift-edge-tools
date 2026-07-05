struct NeedleConfidenceState {
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
