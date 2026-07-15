#if MLX && canImport(MLX)
  import MLX

  // MARK: - EdgeToolsMetadataKey

  extension EdgeToolsMetadataKey {
    public static let mlxEngineGenerationStartMemorySnapshot =
      EdgeToolsMetadataKey(rawValue: "MLXEngineGenerationStartMemorySnapshot")

    public static let mlxEnginePostPrefillMemorySnapshot =
      EdgeToolsMetadataKey(rawValue: "MLXEnginePostPrefillMemorySnapshot")

    public static let mlxEnginePostDecodeMemorySnapshot =
      EdgeToolsMetadataKey(rawValue: "MLXEnginePostDecodeMemorySnapshot")
  }

  // MARK: - EdgeToolsMetadata

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
