public struct NeedlePrefillMetrics: Hashable, Sendable {
  public let tokens: Int
  public let duration: Duration

  public init(tokens: Int, duration: Duration) {
    self.tokens = tokens
    self.duration = duration
  }
}
