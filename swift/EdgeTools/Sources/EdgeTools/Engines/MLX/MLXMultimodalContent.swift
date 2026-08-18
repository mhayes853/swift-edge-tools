#if MLX && canImport(MLXVLM)
  import MLXLMCommon

  extension EdgeToolsMultimodalContent {
    var mlxMessage: MLXLMCommon.Message {
      switch self {
      case .text(let text): ["type": "text", "text": text]
      case .image: ["type": "image"]
      case .video: ["type": "video"]
      case .audio: ["type": "audio"]
      }
    }
  }
#endif
