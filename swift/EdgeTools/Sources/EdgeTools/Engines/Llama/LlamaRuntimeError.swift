#if Llama && canImport(EdgeToolsLlama)
  import EdgeToolsLlama

  extension LlamaRuntimeError.Code {
    public static let contextCreationFailed = Self(rawValue: "context-creation-failed")
    public static let decodeFailed = Self(rawValue: "decode-failed")
    public static let multimodalProjectorLoadFailed =
      Self(rawValue: "multimodal-projector-load-failed")
    public static let multimodalProcessingFailed = Self(rawValue: "multimodal-processing-failed")
  }
#endif
