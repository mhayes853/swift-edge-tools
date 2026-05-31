public struct NeedlePrefillMetrics: Hashable, Sendable {
  public let tokens: Int
  public let duration: Duration
  public let ramUsageBytes: Int64

  public init(tokens: Int, duration: Duration, ramUsageBytes: Int64) {
    self.tokens = tokens
    self.duration = duration
    self.ramUsageBytes = ramUsageBytes
  }
}
