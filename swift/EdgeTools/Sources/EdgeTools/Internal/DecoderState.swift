// MARK: - DecoderState

struct DecoderState<Sampler> {
  var pendingTokenId: EdgeToolsToken.ID?
  let sampler: Sampler
  let confidenceOptions: EdgeToolsConfidenceOptions
  private var confidence = ConfidenceState()

  init(sampler: Sampler, confidenceOptions: EdgeToolsConfidenceOptions) {
    self.sampler = sampler
    self.confidenceOptions = confidenceOptions
  }

  var tracksTokenConfidence: Bool {
    !self.tokenConfidenceOptions.isEmpty
  }

  var metrics: EdgeToolsMetrics {
    var metrics = EdgeToolsMetrics()
    if self.confidenceOptions.contains(.generation) {
      metrics.generationConfidence = self.confidence.mean
    }
    if self.confidenceOptions.contains(.perToken) {
      metrics.perTokenConfidences = self.confidence.perTokenConfidences
    }
    return metrics
  }

  private var tokenConfidenceOptions: EdgeToolsConfidenceOptions {
    self.confidenceOptions.intersection([.generation, .perToken])
  }

  mutating func add(confidence: Float) {
    self.confidence.add(confidence: confidence, options: self.tokenConfidenceOptions)
  }
}
