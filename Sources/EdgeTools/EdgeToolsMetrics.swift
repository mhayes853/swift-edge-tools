// MARK: - EdgeToolsDecodeMetrics

public struct EdgeToolsDecodeMetrics: Hashable, Sendable {
  public let tokens: Int
  public let duration: Duration
  public let durationToFirstToken: Duration

  public init(
    tokens: Int,
    duration: Duration,
    durationToFirstToken: Duration
  ) {
    self.tokens = tokens
    self.duration = duration
    self.durationToFirstToken = durationToFirstToken
  }
}

// MARK: - EdgeToolsPrefillMetrics

public struct EdgeToolsPrefillMetrics: Hashable, Sendable {
  public let tokens: Int
  public let duration: Duration

  public init(tokens: Int, duration: Duration) {
    self.tokens = tokens
    self.duration = duration
  }
}
