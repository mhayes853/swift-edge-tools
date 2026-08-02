import EdgeTools

final class LockBox<Value: Sendable>: Sendable {
  private let storage: Lock<Value>

  init(_ value: sending Value) {
    self.storage = Lock(value)
  }

  func withLock<Result>(_ body: (inout sending Value) -> sending Result) -> sending Result {
    self.storage.withLock(body)
  }
}
