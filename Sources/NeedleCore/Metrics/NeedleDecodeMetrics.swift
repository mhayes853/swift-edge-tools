public struct NeedleDecodeMetrics: Hashable, Sendable {
  public let tokens: Int
  public let duration: Duration
  public let durationToFirstToken: Duration
  public let ramUsageBytes: Int64

  public init(
    tokens: Int,
    duration: Duration,
    durationToFirstToken: Duration,
    ramUsageBytes: Int64
  ) {
    self.tokens = tokens
    self.duration = duration
    self.durationToFirstToken = durationToFirstToken
    self.ramUsageBytes = ramUsageBytes
  }
}
