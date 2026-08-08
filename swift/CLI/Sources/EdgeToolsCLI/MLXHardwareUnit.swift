public enum MLXHardwareUnit: String, CaseIterable, Sendable {
  case cpu
  case gpu

  public init?(argument: String) {
    switch argument.lowercased().filter({ $0.isLetter || $0.isNumber }) {
    case "cpu": self = .cpu
    case "gpu": self = .gpu
    default: return nil
    }
  }
}

#if canImport(MLX)
  import MLX

  extension MLXHardwareUnit {
    var device: Device {
      switch self {
      case .cpu: .cpu
      case .gpu: .gpu
      }
    }

    func withDefaultDevice<R>(_ body: () async throws -> R) async rethrows -> R {
      try await Device.withDefaultDevice(self.device, body)
    }
  }
#else
  extension MLXHardwareUnit {
    func withDefaultDevice<R>(_ body: () async throws -> R) async rethrows -> R {
      try await body()
    }
  }
#endif
