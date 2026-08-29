#if Llama && canImport(CLlama) && !os(WASI)
  import EdgeTools
  import Foundation

  func llamaTestModelParameters() -> LlamaModelParameters {
    LlamaModelParameters(gpuLayerCount: isLlamaGPUDisabled() ? 0 : .max)
  }

  func llamaTestMultimodalParameters(warmUp: Bool = true) -> LlamaMultimodalParameters {
    LlamaMultimodalParameters(useGPU: !isLlamaGPUDisabled(), warmUp: warmUp)
  }

  private func isLlamaGPUDisabled() -> Bool {
    guard let value = ProcessInfo.processInfo.environment["EDGE_TOOLS_DISABLE_LLAMA_GPU"] else {
      return false
    }
    let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !normalized.isEmpty else { return false }
    return normalized != "0" && normalized != "false"
  }
#endif
