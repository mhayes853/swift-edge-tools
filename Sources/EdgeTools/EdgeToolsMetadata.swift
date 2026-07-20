#if MLX && canImport(MLX)
  import MLX
#endif

// MARK: - EdgeToolsMetadataKey

public struct EdgeToolsMetadataKey: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral
{
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: StringLiteralType) {
    self.init(rawValue: value)
  }
}

// MARK: - EdgeToolsMetadataKey + Confidence

extension EdgeToolsMetadataKey {
  public static let generationConfidence =
    EdgeToolsMetadataKey(rawValue: "GenerationConfidence")

  public static let perTokenConfidences =
    EdgeToolsMetadataKey(rawValue: "PerTokenConfidences")
}

// MARK: - EdgeToolsMetadata

public typealias EdgeToolsMetadata = [EdgeToolsMetadataKey: any Sendable]

// MARK: - EdgeToolsMetadata + Confidence

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

#if MLX && canImport(MLX)
  // MARK: - EdgeToolsMetadataKey + MLX

  extension EdgeToolsMetadataKey {
    public static let mlxEngineGenerationStartMemorySnapshot =
      EdgeToolsMetadataKey(rawValue: "MLXEngineGenerationStartMemorySnapshot")

    public static let mlxEnginePostPrefillMemorySnapshot =
      EdgeToolsMetadataKey(rawValue: "MLXEnginePostPrefillMemorySnapshot")

    public static let mlxEnginePostDecodeMemorySnapshot =
      EdgeToolsMetadataKey(rawValue: "MLXEnginePostDecodeMemorySnapshot")
  }

  // MARK: - EdgeToolsMetadata + MLX

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
