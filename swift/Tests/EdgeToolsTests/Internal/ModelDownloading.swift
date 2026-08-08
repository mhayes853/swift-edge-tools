import Foundation

#if MLX
  import Hub

  func downloadNeedle() async throws -> URL {
    try await downloadModel(id: "Cactus-Compute/needle")
  }

  func downloadQwen3() async throws -> URL {
    try await downloadModel(id: "mlx-community/Qwen3-0.6B-4bit")
  }

  func downloadQwen3P5() async throws -> URL {
    try await downloadModel(id: "mlx-community/Qwen3.5-0.8B-MLX-4bit")
  }

  func downloadQwen3P5VL() async throws -> URL {
    try await downloadModel(id: "mlx-community/Qwen3.5-2B-4bit")
  }

  func downloadFunctionGemma() async throws -> URL {
    try await downloadModel(id: "mlx-community/functiongemma-270m-it-4bit")
  }

  func downloadLFM2P5() async throws -> URL {
    try await downloadModel(id: "LiquidAI/LFM2.5-230M-MLX-4bit")
  }

  func downloadLFM2P5VL() async throws -> URL {
    try await downloadModel(id: "LiquidAI/LFM2.5-VL-450M-MLX-4bit")
  }

  func downloadGemma4E2B() async throws -> URL {
    try await downloadModel(id: "mlx-community/gemma-4-e2b-it-4bit")
  }

  func downloadMiniCPM5() async throws -> URL {
    try await downloadModel(id: "openbmb/MiniCPM5-1B-MLX")
  }

  func downloadGraniteMoeHybrid() async throws -> URL {
    try await downloadModel(id: "mlx-community/granite-4.0-h-350m-5bit")
  }

  private func downloadModel(id: String) async throws -> URL {
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: id, type: .models)
    let destination = hub.localRepoLocation(repo)

    if hasCompatibleTokenizer(in: destination) {
      print("=== Model Already Downloaded At \(destination.path) ===")
      return destination
    }

    print("=== Downloading Model From \(repo.id) ===")
    let url = try await hub.snapshot(from: repo)
    print("=== Finished Downloading Model To \(url.path) ===")
    return url
  }

  private func hasCompatibleTokenizer(in directory: URL) -> Bool {
    ["tokenizer.model", "tokenizer.json"]
      .contains {
        FileManager.default.fileExists(atPath: directory.appending(path: $0).path())
      }
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
