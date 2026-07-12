#if canImport(Darwin) && canImport(Foundation)
  import Foundation
#endif

#if canImport(Synchronization)
  import Synchronization
#endif

package struct Lock<Value: ~Copyable>: ~Copyable {
  #if canImport(Darwin) && canImport(Foundation)
    private let lock = NSLock()
    private var value: UnsafeMutablePointer<Value>
  #else
    private let lock: Mutex<Value>
  #endif

  package init(_ value: consuming sending Value) {
    #if canImport(Darwin) && canImport(Foundation)
      self.value = UnsafeMutablePointer<Value>.allocate(capacity: 1)
      self.value.initialize(to: value)
    #else
      self.lock = Mutex(value)
    #endif
  }

  deinit {
    #if canImport(Darwin) && canImport(Foundation)
      self.value.deinitialize(count: 1)
      self.value.deallocate()
    #endif
  }

  package borrowing func withLock<Result: ~Copyable, E: Error>(
    _ body: (inout sending Value) throws(E) -> sending Result
  ) throws(E) -> sending Result {
    #if canImport(Darwin) && canImport(Foundation)
      self.lock.lock()
      defer { self.lock.unlock() }
      return try body(&self.value.pointee)
    #else
      try self.lock.withLock(body)
    #endif
  }
}

extension Lock: @unchecked Sendable where Value: ~Copyable {}
