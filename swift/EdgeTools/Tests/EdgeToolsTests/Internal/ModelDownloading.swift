import Foundation

#if HuggingFaceTokenizers
  import Hub

  enum ModelID: String {
    case qwen3 = "mlx-community/Qwen3-0.6B-4bit"
    case qwen3P5 = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    case qwen3P5VL = "mlx-community/Qwen3.5-2B-4bit"
    case functionGemma = "mlx-community/functiongemma-270m-it-4bit"
    case lfm2P5 = "LiquidAI/LFM2.5-230M-MLX-4bit"
    case lfm2P5VL = "LiquidAI/LFM2.5-VL-450M-MLX-4bit"
    case lfm2P5Thinking = "LiquidAI/LFM2.5-1.2B-Thinking-MLX-4bit"
    case gemma4E2B = "mlx-community/gemma-4-e2b-it-4bit"
    case miniCPM5 = "openbmb/MiniCPM5-1B-MLX"
    case graniteMoeHybrid = "mlx-community/granite-4.0-h-350m-5bit"
  }

  func downloadModel(id: ModelID) async throws -> URL {
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: id.rawValue, type: .models)
    let destination = hub.localRepoLocation(repo)

    if FileManager.default.fileExists(atPath: destination.appending(path: "tokenizer.json").path()) {
      print("=== Model Already Downloaded At \(destination.path) ===")
      return destination
    }

    print("=== Downloading Model From \(repo.id) ===")
    let url = try await hub.snapshot(from: repo)
    print("=== Finished Downloading Model To \(url.path) ===")
    return url
  }
#endif

#if MLX

  func downloadQwen3() async throws -> URL {
    try await downloadModel(id: .qwen3)
  }

  func downloadQwen3P5() async throws -> URL {
    try await downloadModel(id: .qwen3P5)
  }

  func downloadQwen3P5VL() async throws -> URL {
    try await downloadModel(id: .qwen3P5VL)
  }

  func downloadFunctionGemma() async throws -> URL {
    try await downloadModel(id: .functionGemma)
  }

  func downloadLFM2P5() async throws -> URL {
    try await downloadModel(id: .lfm2P5)
  }

  func downloadLFM2P5VL() async throws -> URL {
    try await downloadModel(id: .lfm2P5VL)
  }

  func downloadLFM2P5Thinking() async throws -> URL {
    try await downloadModel(id: .lfm2P5Thinking)
  }

  func downloadGemma4E2B() async throws -> URL {
    try await downloadModel(id: .gemma4E2B)
  }

  func downloadMiniCPM5() async throws -> URL {
    try await downloadModel(id: .miniCPM5)
  }

  func downloadGraniteMoeHybrid() async throws -> URL {
    try await downloadModel(id: .graniteMoeHybrid)
  }

#endif

#if HuggingFaceTokenizers && Llama && canImport(CLlama)

  enum GGUFModelID: String {
    case qwen3 = "Qwen/Qwen3-0.6B-GGUF"
    case qwen3P5 = "unsloth/Qwen3.5-0.8B-GGUF"
    case functionGemma = "ggml-org/functiongemma-270m-it-GGUF"
    case lfm2P5 = "LiquidAI/LFM2.5-230M-GGUF"
    case lfm2P5Thinking = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF"
    case gemma4E2BHybrid = "Cactus-Compute/gemma-4-e2b-it-hybrid-GGUF"
    case miniCPM5 = "openbmb/MiniCPM5-1B-GGUF"
    case graniteMoeHybrid = "ibm-granite/granite-4.0-h-350m-GGUF"

    var file: String {
      switch self {
      case .qwen3: "Qwen3-0.6B-Q8_0.gguf"
      case .qwen3P5: "Qwen3.5-0.8B-Q4_K_M.gguf"
      case .functionGemma: "functiongemma-270m-it-q8_0.gguf"
      case .lfm2P5: "LFM2.5-230M-Q8_0.gguf"
      case .lfm2P5Thinking: "LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
      case .gemma4E2BHybrid: "gemma-4-e2b-it-hybrid-Q4_K_M.gguf"
      case .miniCPM5: "MiniCPM5-1B-Q4_K_M.gguf"
      case .graniteMoeHybrid: "granite-4.0-h-350m-Q4_K_M.gguf"
      }
    }
  }

  func downloadGGUFModel(id: GGUFModelID) async throws -> URL {
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: id.rawValue, type: .models)
    let destination = hub.localRepoLocation(repo).appending(path: id.file)

    if FileManager.default.fileExists(atPath: destination.path()) {
      print("=== GGUF Already Downloaded At \(destination.path) ===")
      return destination
    }

    print("=== Downloading GGUF From \(repo.id) ===")
    let url = try await hub.snapshot(from: repo, matching: [id.file])
    print("=== Finished Downloading GGUF To \(url.path) ===")
    return url.appending(path: id.file)
  }
#endif

extension URL {
  static let swiftEdgeToolsTestsDirectory = {
    #if os(macOS) || os(Linux)
      URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".edge-tools-tests")
    #elseif canImport(Darwin)
      URL.documentsDirectory
        .appendingPathComponent(".edge-tools-tests")
    #else
      FileManager.default.temporaryDirectory
        .appendingPathComponent(".edge-tools-tests")
    #endif
  }()
}
