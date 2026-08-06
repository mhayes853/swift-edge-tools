import Darwin
import EdgeTools
import Foundation
import MLX

// MARK: - RunMetrics

public struct RunMetrics: Sendable {
  public var loadDuration: Duration
  public var generationDuration: Duration
  public var prefill: EdgeToolsPrefillMetrics
  public var decode: EdgeToolsDecodeMetrics
  public var peakResidentBytes: Int
  public var peakGPUBytes: Int

  public init(
    loadDuration: Duration,
    generationDuration: Duration,
    prefill: EdgeToolsPrefillMetrics,
    decode: EdgeToolsDecodeMetrics,
    peakResidentBytes: Int,
    peakGPUBytes: Int
  ) {
    self.loadDuration = loadDuration
    self.generationDuration = generationDuration
    self.prefill = prefill
    self.decode = decode
    self.peakResidentBytes = peakResidentBytes
    self.peakGPUBytes = peakGPUBytes
  }
}

// MARK: - Sampling

public func peakResidentBytes() -> Int {
  var info = rusage()
  guard getrusage(RUSAGE_SELF, &info) == 0 else { return 0 }
  return Int(info.ru_maxrss)
}

public func peakGPUBytes() -> Int {
  GPU.snapshot().peakMemory
}

// MARK: - Formatting

extension Duration {
  public var milliseconds: Double {
    Double(self.components.seconds) * 1000
      + Double(self.components.attoseconds) / 1e15
  }

  public var formattedDuration: String {
    let milliseconds = self.milliseconds
    return milliseconds >= 1000
      ? String(format: "%.2fs", milliseconds / 1000)
      : String(format: "%.0fms", milliseconds)
  }
}

public func formattedBytes(_ bytes: Int) -> String {
  let megabytes = Double(bytes) / 1_048_576
  return megabytes >= 1024
    ? String(format: "%.2f GB", megabytes / 1024)
    : String(format: "%.0f MB", megabytes)
}
