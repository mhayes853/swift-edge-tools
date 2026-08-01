#if canImport(Darwin) && canImport(os)
  import os
#endif

#if canImport(Synchronization)
  import Synchronization
#endif

package struct Lock<Value: ~Copyable>: ~Copyable {
  #if canImport(Darwin) && canImport(os)
    private let lock = OSAllocatedUnfairLock()
    private var value: UnsafeMutablePointer<Value>
  #else
    private let lock: Mutex<Value>
  #endif

  package init(_ value: consuming sending Value) {
    #if canImport(Darwin) && canImport(os)
      self.value = UnsafeMutablePointer<Value>.allocate(capacity: 1)
      self.value.initialize(to: value)
    #else
      self.lock = Mutex(value)
    #endif
  }

  package init<E: Error>(_ makeValue: () throws(E) -> Value) throws(E) {
    #if canImport(Darwin) && canImport(os)
      let value = try makeValue()
      self.value = UnsafeMutablePointer<Value>.allocate(capacity: 1)
      self.value.initialize(to: value)
    #else
      self.lock = Mutex(try makeValue())
    #endif
  }

  deinit {
    #if canImport(Darwin) && canImport(os)
      self.value.deinitialize(count: 1)
      self.value.deallocate()
    #endif
  }

  package borrowing func withLock<Result: ~Copyable, E: Error>(
    _ body: (inout sending Value) throws(E) -> sending Result
  ) throws(E) -> sending Result {
    #if canImport(Darwin) && canImport(os)
      self.lock.lock()
      defer { self.lock.unlock() }
      return try body(&self.value.pointee)
    #else
      try self.lock.withLock(body)
    #endif
  }

  package borrowing func withBorrowedLock<Result: ~Copyable, E: Error>(
    _ body: (borrowing Value) throws(E) -> sending Result
  ) throws(E) -> sending Result {
    #if canImport(Darwin) && canImport(os)
      self.lock.lock()
      defer { self.lock.unlock() }
      return try body(self.value.pointee)
    #else
      try self.lock.withLock {
        (value: inout sending Value) throws(E) -> sending Result in
        try body(value)
      }
    #endif
  }
}

extension Lock: @unchecked Sendable where Value: ~Copyable {}
