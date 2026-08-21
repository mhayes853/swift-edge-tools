#if MLX && canImport(MLX)
  import MLX
#endif

// MARK: - EdgeToolsMetricKey

public struct EdgeToolsMetricKey:
    Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: String) {
    self.init(rawValue: value)
  }
}

// MARK: - EdgeToolsMetrics

public typealias EdgeToolsMetrics = [EdgeToolsMetricKey: any Sendable]

// MARK: - Prefill/Decode

extension EdgeToolsMetricKey {
  public static let prefillTokens = Self(rawValue: "PrefillTokens")
  public static let prefillDuration = Self(rawValue: "PrefillDuration")
  public static let prefillTokensPerSecond = Self(rawValue: "PrefillTokensPerSecond")
  public static let decodeTokens = Self(rawValue: "DecodeTokens")
  public static let decodeDuration = Self(rawValue: "DecodeDuration")
  public static let decodeTokensPerSecond = Self(rawValue: "DecodeTokensPerSecond")
  public static let durationToFirstToken = Self(rawValue: "DurationToFirstToken")
}

extension EdgeToolsMetrics {
  public var prefillTokens: Int? {
    get { self[.prefillTokens] as? Int }
    set { self[.prefillTokens] = newValue }
  }

  public var prefillDuration: Duration? {
    get { self[.prefillDuration] as? Duration }
    set { self[.prefillDuration] = newValue }
  }

  public var prefillTokensPerSecond: Double? {
    get {
      if let rate = self[.prefillTokensPerSecond] as? Double { return rate }
      guard let tokens = self.prefillTokens, let duration = self.prefillDuration else { return nil }
      return tokensPerSecond(tokens: tokens, duration: duration)
    }
    set { self[.prefillTokensPerSecond] = newValue }
  }

  public var decodeTokens: Int? {
    get { self[.decodeTokens] as? Int }
    set { self[.decodeTokens] = newValue }
  }

  public var decodeDuration: Duration? {
    get { self[.decodeDuration] as? Duration }
    set { self[.decodeDuration] = newValue }
  }

  public var decodeTokensPerSecond: Double? {
    get {
      if let rate = self[.decodeTokensPerSecond] as? Double { return rate }
      guard let tokens = self.decodeTokens, let duration = self.decodeDuration else { return nil }
      return tokensPerSecond(tokens: tokens, duration: duration)
    }
    set { self[.decodeTokensPerSecond] = newValue }
  }

  public var durationToFirstToken: Duration? {
    get { self[.durationToFirstToken] as? Duration }
    set { self[.durationToFirstToken] = newValue }
  }
}

// MARK: - Confidence

extension EdgeToolsMetricKey {
  public static let generationConfidence = Self(rawValue: "GenerationConfidence")
  public static let perTokenConfidences = Self(rawValue: "PerTokenConfidences")
  public static let probeConfidence = Self(rawValue: "ProbeConfidence")
}

extension EdgeToolsMetrics {
  public var generationConfidence: Float? {
    get { self[.generationConfidence] as? Float }
    set { self[.generationConfidence] = newValue }
  }

  public var perTokenConfidences: [Float]? {
    get { self[.perTokenConfidences] as? [Float] }
    set { self[.perTokenConfidences] = newValue }
  }

  public var probeConfidence: Float? {
    get { self[.probeConfidence] as? Float }
    set { self[.probeConfidence] = newValue }
  }
}

// MARK: - MLX

#if MLX && canImport(MLX)
  extension EdgeToolsMetricKey {
    public static let mlxEngineGenerationStartMemorySnapshot =
      Self(rawValue: "MLXEngineGenerationStartMemorySnapshot")

    public static let mlxEnginePostPrefillMemorySnapshot =
      Self(rawValue: "MLXEnginePostPrefillMemorySnapshot")

    public static let mlxEnginePostDecodeMemorySnapshot =
      Self(rawValue: "MLXEnginePostDecodeMemorySnapshot")
  }

  extension EdgeToolsMetrics {
    public var mlxEngineGenerationStartMemorySnapshot: Memory.Snapshot? {
      get { self[.mlxEngineGenerationStartMemorySnapshot] as? Memory.Snapshot }
      set { self[.mlxEngineGenerationStartMemorySnapshot] = newValue }
    }

    public var mlxEnginePostPrefillMemorySnapshot: Memory.Snapshot? {
      get { self[.mlxEnginePostPrefillMemorySnapshot] as? Memory.Snapshot }
      set { self[.mlxEnginePostPrefillMemorySnapshot] = newValue }
    }

    public var mlxEnginePostDecodeMemorySnapshot: Memory.Snapshot? {
      get { self[.mlxEnginePostDecodeMemorySnapshot] as? Memory.Snapshot }
      set { self[.mlxEnginePostDecodeMemorySnapshot] = newValue }
    }
  }
#endif

private func tokensPerSecond(tokens: Int, duration: Duration) -> Double {
  Double(tokens) / (
    Double(duration.components.seconds)
      + Double(duration.components.attoseconds) / attosecondsPerSecond
  )
}

private let attosecondsPerSecond = 1e18
