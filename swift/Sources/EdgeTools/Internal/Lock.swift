#if canImport(Darwin) && canImport(os)
  import os
#endif

#if $Embedded
  #if _runtime(_multithreaded)
    import Synchronization
  #endif
#elseif canImport(Synchronization)
  import Synchronization
#endif

package struct Lock<Value: ~Copyable>: ~Copyable {
  #if $Embedded
    #if _runtime(_multithreaded)
      // TODO: - Do we need to do a spin lock here?
      private let lock = Atomic<Bool>(false)
    #endif
    private var value: UnsafeMutablePointer<Value>
  #elseif canImport(Darwin) && canImport(os)
    private let lock = OSAllocatedUnfairLock()
    private var value: UnsafeMutablePointer<Value>
  #else
    private let lock: Mutex<Value>
  #endif

  package init(_ value: consuming sending Value) {
    #if $Embedded || (canImport(Darwin) && canImport(os))
      self.value = UnsafeMutablePointer<Value>.allocate(capacity: 1)
      self.value.initialize(to: value)
    #else
      self.lock = Mutex(value)
    #endif
  }

  package init<E: Error>(_ makeValue: () throws(E) -> Value) throws(E) {
    #if $Embedded || (canImport(Darwin) && canImport(os))
      let value = try makeValue()
      self.value = UnsafeMutablePointer<Value>.allocate(capacity: 1)
      self.value.initialize(to: value)
    #else
      self.lock = Mutex(try makeValue())
    #endif
  }

  deinit {
    #if $Embedded || (canImport(Darwin) && canImport(os))
      self.value.deinitialize(count: 1)
      self.value.deallocate()
    #endif
  }

  package borrowing func withLock<Result: ~Copyable, E: Error>(
    _ body: (inout sending Value) throws(E) -> sending Result
  ) throws(E) -> sending Result {
    #if $Embedded
      self.acquire()
      defer { self.release() }
      return try body(&self.value.pointee)
    #elseif canImport(Darwin) && canImport(os)
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
    #if $Embedded
      self.acquire()
      defer { self.release() }
      return try body(self.value.pointee)
    #elseif canImport(Darwin) && canImport(os)
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

  #if $Embedded
    private borrowing func acquire() {
      #if _runtime(_multithreaded)
        while !self.lock.compareExchange(
          expected: false,
          desired: true,
          ordering: .acquiring
        ).exchanged {}
      #endif
    }

    private borrowing func release() {
      #if _runtime(_multithreaded)
        self.lock.store(false, ordering: .releasing)
      #endif
    }
  #endif
}

extension Lock: @unchecked Sendable where Value: ~Copyable {}
