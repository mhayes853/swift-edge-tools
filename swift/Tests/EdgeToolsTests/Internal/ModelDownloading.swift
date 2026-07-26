import Foundation

#if canImport(Hub)
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

  func downloadFunctionGemma() async throws -> URL {
    try await downloadModel(id: "mlx-community/functiongemma-270m-it-4bit")
  }

  func downloadLFM2() async throws -> URL {
    try await downloadModel(id: "LiquidAI/LFM2.5-230M-MLX-4bit")
  }

  private func downloadModel(id: String) async throws -> URL {
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: id, type: .models)
    let destination = hub.localRepoLocation(repo)

    if FileManager.default.fileExists(atPath: destination.path) {
      print("=== Model Already Downloaded At \(destination.path) ===")
      return destination
    }

    print("=== Downloading Model From \(repo.id) ===")
    let url = try await hub.snapshot(from: repo)
    print("=== Finished Downloading Model To \(url.path) ===")
    return url
  }
#endif

extension URL {
  static let swiftEdgeToolsTestsDirectory = {
    #if os(macOS) || os(Linux)
      URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".swift-needle-tests")
    #elseif canImport(Darwin)
      URL.documentsDirectory
        .appendingPathComponent(".swift-needle-tests")
    #else
      FileManager.default.temporaryDirectory
        .appendingPathComponent(".swift-needle-tests")
    #endif
  }()
}
