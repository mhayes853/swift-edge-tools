#if MLX && canImport(MLX)
  import MLX
#endif

// MARK: - EdgeToolsMetadataKey

public struct EdgeToolsMetadataKey:
    Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: String) {
    self.init(rawValue: value)
  }
}

// MARK: - EdgeToolsMetadata

public typealias EdgeToolsMetadata = [EdgeToolsMetadataKey: any Sendable]

// MARK: - Confidence

extension EdgeToolsMetadataKey {
  public static let generationConfidence = Self(rawValue: "GenerationConfidence")
  public static let perTokenConfidences = Self(rawValue: "PerTokenConfidences")
}

extension EdgeToolsMetadata {
  public var generationConfidence: Float? {
    get { self[.generationConfidence] as? Float }
    set { self[.generationConfidence] = newValue }
  }

  public var perTokenConfidences: [Float]? {
    get { self[.perTokenConfidences] as? [Float] }
    set { self[.perTokenConfidences] = newValue }
  }
}

// MARK: - MLX

#if MLX && canImport(MLX)
  extension EdgeToolsMetadataKey {
    public static let mlxEngineGenerationStartMemorySnapshot =
      Self(rawValue: "MLXEngineGenerationStartMemorySnapshot")

    public static let mlxEnginePostPrefillMemorySnapshot =
      Self(rawValue: "MLXEnginePostPrefillMemorySnapshot")

    public static let mlxEnginePostDecodeMemorySnapshot =
      Self(rawValue: "MLXEnginePostDecodeMemorySnapshot")
  }

  extension EdgeToolsMetadata {
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
