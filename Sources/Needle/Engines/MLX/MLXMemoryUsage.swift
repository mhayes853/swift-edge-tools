#if SwiftNeedleMLX
  import MLX

  // MARK: - MemoryUsage

  struct MLXMemoryUsage: ~Copyable {
    private let start: Memory.Snapshot
    private let stream: Stream

    init() {
      self.stream = Stream.defaultStream(.defaultDevice())
      self.stream.synchronize()
      self.start = Memory.snapshot()
      Memory.peakMemory = start.activeMemory
    }

    consuming func stop() -> Int64 {
      self.stream.synchronize()
      return Int64(Memory.peakMemory - self.start.activeMemory)
    }
  }

  // MARK: - With Helpers

  func withMemoryUsage<R>(_ body: () throws -> R) rethrows -> (R, Int64) {
    let usage = MLXMemoryUsage()
    let result = try body()
    return (result, usage.stop())
  }

  func withMemoryUsage(_ body: () throws -> Void) rethrows -> Int64 {
    try withMemoryUsage(body).1
  }
#endif
