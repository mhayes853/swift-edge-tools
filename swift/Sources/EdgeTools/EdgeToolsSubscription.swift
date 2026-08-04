public final class EdgeToolsSubscription: Sendable {
  private let cancellation: Lock<(@Sendable () -> Void)?>

  public init(_ cancellation: @escaping @Sendable () -> Void) {
    self.cancellation = Lock(cancellation)
  }

  deinit { self.cancel() }

  public func cancel() {
    let cancellation = self.cancellation.withLock { cancellation in
      defer { cancellation = nil }
      return cancellation
    }
    cancellation?()
  }
}
