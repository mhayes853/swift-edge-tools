// MARK: - EdgeToolsDecodeMetrics

public struct EdgeToolsDecodeMetrics: Hashable, Sendable {
  public let tokens: Int
  public let duration: Duration
  public let durationToFirstToken: Duration

  public var tokensPerSecond: Double {
    EdgeTools.tokensPerSecond(tokens: self.tokens, duration: self.duration)
  }

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

  public var tokensPerSecond: Double {
    EdgeTools.tokensPerSecond(tokens: self.tokens, duration: self.duration)
  }

  public init(tokens: Int, duration: Duration) {
    self.tokens = tokens
    self.duration = duration
  }
}

private func tokensPerSecond(tokens: Int, duration: Duration) -> Double {
  Double(tokens) / (
    Double(duration.components.seconds)
      + Double(duration.components.attoseconds) / attosecondsPerSecond
  )
}

private let attosecondsPerSecond = 1e18
