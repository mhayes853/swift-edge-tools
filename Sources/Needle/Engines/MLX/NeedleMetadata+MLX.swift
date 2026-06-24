#if SwiftNeedleMLX
  import MLX

  // MARK: - NeedleMetadataKey (MLX)

  extension NeedleMetadataKey {
    public static let mlxEngineGenerationStartMemorySnapshot =
      NeedleMetadataKey(rawValue: "MLXEngineGenerationStartMemorySnapshot")

    public static let mlxEnginePostPrefillMemorySnapshot =
      NeedleMetadataKey(rawValue: "MLXEnginePostPrefillMemorySnapshot")

    public static let mlxEnginePostDecodeMemorySnapshot =
      NeedleMetadataKey(rawValue: "MLXEnginePostDecodeMemorySnapshot")

    public static let mlxEngineGenerationConfidence =
      NeedleMetadataKey(rawValue: "MLXEngineGenerationConfidence")

    public static let mlxEngineTokenUncertainties =
      NeedleMetadataKey(rawValue: "MLXEngineTokenUncertainties")
  }

  // MARK: - NeedleMetadata (MLX)

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

    public var mlxEngineTokenUncertainties: [Float]? {
      get { self[.mlxEngineTokenUncertainties] as? [Float] }
      set { self[.mlxEngineTokenUncertainties] = newValue }
    }
  }
#endif
