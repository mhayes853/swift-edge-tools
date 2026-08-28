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
    let url = try await retryingTransientNetworkFailures { try await hub.snapshot(from: repo) }
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

  private enum GGUFModelDownloadError: Error {
    case missingMultimodalProjector
  }

  enum GGUFModelID: String {
    case qwen3 = "Qwen/Qwen3-0.6B-GGUF"
    case qwen3P5 = "unsloth/Qwen3.5-0.8B-GGUF"
    case qwen3P5VL = "bartowski/Qwen_Qwen3.5-2B-GGUF"
    case functionGemma = "ggml-org/functiongemma-270m-it-GGUF"
    case lfm2P5 = "LiquidAI/LFM2.5-230M-GGUF"
    case lfm2P5VL = "LiquidAI/LFM2.5-VL-450M-GGUF"
    case lfm2P5Thinking = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF"
    case gemma4E2BHybrid = "Cactus-Compute/gemma-4-e2b-it-hybrid-GGUF"
    case miniCPM5 = "openbmb/MiniCPM5-1B-GGUF"
    case graniteMoeHybrid = "ibm-granite/granite-4.0-h-350m-GGUF"

    var file: String {
      switch self {
      case .qwen3: "Qwen3-0.6B-Q8_0.gguf"
      case .qwen3P5: "Qwen3.5-0.8B-Q4_K_M.gguf"
      case .qwen3P5VL: "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
      case .functionGemma: "functiongemma-270m-it-q8_0.gguf"
      case .lfm2P5: "LFM2.5-230M-Q8_0.gguf"
      case .lfm2P5VL: "LFM2.5-VL-450M-Q8_0.gguf"
      case .lfm2P5Thinking: "LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
      case .gemma4E2BHybrid: "gemma-4-e2b-it-hybrid-Q4_K_M.gguf"
      case .miniCPM5: "MiniCPM5-1B-Q4_K_M.gguf"
      case .graniteMoeHybrid: "granite-4.0-h-350m-Q4_K_M.gguf"
      }
    }

    var multimodalProjectorFile: String? {
      switch self {
      case .gemma4E2BHybrid: "mmproj-F16.gguf"
      case .qwen3P5VL: "mmproj-Qwen_Qwen3.5-2B-f16.gguf"
      case .lfm2P5VL: "mmproj-LFM2.5-VL-450m-Q8_0.gguf"
      case .qwen3, .qwen3P5, .functionGemma, .lfm2P5, .lfm2P5Thinking, .miniCPM5,
        .graniteMoeHybrid:
        nil
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
    let url = try await retryingTransientNetworkFailures {
      try await hub.snapshot(from: repo, matching: [id.file])
    }
    print("=== Finished Downloading GGUF To \(url.path) ===")
    return url.appending(path: id.file)
  }

  func downloadGGUFMultimodalModel(
    id: GGUFModelID
  ) async throws -> (model: URL, projector: URL) {
    guard let projectorFile = id.multimodalProjectorFile else {
      throw GGUFModelDownloadError.missingMultimodalProjector
    }
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: id.rawValue, type: .models)
    let directory = hub.localRepoLocation(repo)
    let model = directory.appending(path: id.file)
    let projector = directory.appending(path: projectorFile)
    if FileManager.default.fileExists(atPath: model.path()),
      FileManager.default.fileExists(atPath: projector.path())
    {
      print("=== GGUF VLM Already Downloaded At \(directory.path) ===")
      return (model, projector)
    }

    print("=== Downloading GGUF VLM From \(repo.id) ===")
    let url = try await retryingTransientNetworkFailures {
      try await hub.snapshot(from: repo, matching: [id.file, projectorFile])
    }
    print("=== Finished Downloading GGUF VLM To \(url.path) ===")
    return (url.appending(path: id.file), url.appending(path: projectorFile))
  }
#endif

// MARK: - Retrying Downloads

#if HuggingFaceTokenizers
  private func retryingTransientNetworkFailures<Value>(
    _ download: () async throws -> Value
  ) async throws -> Value {
    for attempt in 1..<maximumDownloadAttempts {
      do {
        return try await download()
      } catch let error where isTransientNetworkFailure(error) {
        print("=== Download Attempt \(attempt) Failed With \(error), Retrying ===")
        try await Task.sleep(for: .seconds(attempt * 5))
      }
    }
    return try await download()
  }

  private func isTransientNetworkFailure(_ error: any Error) -> Bool {
    let error = error as NSError
    guard error.domain == NSURLErrorDomain else {
      return false
    }
    return transientNetworkErrorCodes.contains(error.code)
  }

  private let maximumDownloadAttempts = 3

  private let transientNetworkErrorCodes: Set<Int> = [
    NSURLErrorNetworkConnectionLost,
    NSURLErrorTimedOut,
    NSURLErrorCannotConnectToHost,
    NSURLErrorNotConnectedToInternet,
    NSURLErrorDNSLookupFailed,
    NSURLErrorResourceUnavailable
  ]
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
