import Foundation
#if canImport(MLX)
  import MLX
#endif

// MARK: - PeakMemory

public struct PeakMemory: Hashable, Sendable {
  public var resident: MemoryByteCount
  public var gpu: MemoryByteCount

  public static let zero = Self(resident: MemoryByteCount(bytes: 0), gpu: MemoryByteCount(bytes: 0))

  public static var current: Self {
    Self(resident: .peakResident, gpu: .peakGPU)
  }
}

// MARK: - MemoryByteCount

public struct MemoryByteCount: Hashable, Sendable {
  public var bytes: Int
}

extension MemoryByteCount {
  public static var peakResident: Self {
    var info = rusage()
    guard getrusage(RUSAGE_SELF, &info) == 0 else { return Self(bytes: 0) }
    #if os(Linux) || os(Android)
      return Self(bytes: Int(info.ru_maxrss) * 1024)
    #else
      return Self(bytes: Int(info.ru_maxrss))
    #endif
  }

  public static var peakGPU: Self {
    #if canImport(MLX)
      Self(bytes: Memory.snapshot().peakMemory)
    #else
      Self(bytes: 0)
    #endif
  }

  public var isEmpty: Bool { self.bytes <= 0 }

  public var displayText: String {
    let megabytes = Double(self.bytes) / 1_048_576
    return megabytes >= 1024
      ? String(format: "%.2f GB", megabytes / 1024)
      : String(format: "%.0f MB", megabytes)
  }
}

extension MemoryByteCount: Codable {
  public init(from decoder: any Decoder) throws {
    self.bytes = try decoder.singleValueContainer().decode(Int.self)
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(self.bytes)
  }
}
