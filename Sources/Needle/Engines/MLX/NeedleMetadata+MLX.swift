#if MLX
  import MLX

  // MARK: - NeedleMetadataKey

  extension NeedleMetadataKey {
    public static let mlxEngineGenerationStartMemorySnapshot =
      NeedleMetadataKey(rawValue: "MLXEngineGenerationStartMemorySnapshot")

    public static let mlxEnginePostPrefillMemorySnapshot =
      NeedleMetadataKey(rawValue: "MLXEnginePostPrefillMemorySnapshot")

    public static let mlxEnginePostDecodeMemorySnapshot =
      NeedleMetadataKey(rawValue: "MLXEnginePostDecodeMemorySnapshot")

    public static let mlxEngineGenerationConfidence =
      NeedleMetadataKey(rawValue: "MLXEngineGenerationConfidence")

    public static let mlxEnginePerTokenConfidences =
      NeedleMetadataKey(rawValue: "MLXEnginePerTokenConfidences")
  }

  // MARK: - NeedleMetadata

  extension NeedleMetadata {
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

    public var mlxEngineGenerationConfidence: Float? {
      get { self[.mlxEngineGenerationConfidence] as? Float }
      set { self[.mlxEngineGenerationConfidence] = newValue }
    }

    public var mlxEnginePerTokenConfidences: [Float]? {
      get { self[.mlxEnginePerTokenConfidences] as? [Float] }
      set { self[.mlxEnginePerTokenConfidences] = newValue }
    }
  }
#endif
