import Foundation

final class Lock<Value: ~Copyable>: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Value

  init(_ value: consuming sending Value) {
    self.value = value
  }

  borrowing func withLock<Result: ~Copyable, E: Error>(
    _ body: (inout sending Value) throws(E) -> sending Result
  ) throws(E) -> sending Result {
    self.lock.lock()
    defer { self.lock.unlock() }
    return try body(&self.value)
  }
}
