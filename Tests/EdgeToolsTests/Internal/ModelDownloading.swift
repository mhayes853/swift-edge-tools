import Foundation

#if canImport(Hub)
  import Hub

  func downloadNeedleHF() async throws -> URL {
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: "Cactus-Compute/needle-hf", type: .models)
    let destination = hub.localRepoLocation(repo)

    if FileManager.default.fileExists(atPath: destination.path) {
      print("=== Needle HF Model Already Downloaded At \(destination.path) ===")
      return destination
    }

    print("=== Downloading Needle HF Model From \(repo.id) ===")
    let url = try await hub.snapshot(from: repo)
    print("=== Finished Downloading Needle HF Model To \(url.path) ===")
    return url
  }
#endif

extension URL {
  static let swiftEdgeToolsTestsDirectory = {
    #if os(macOS)
      URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".swift-needle-tests")
    #else
      URL.documentsDirectory
        .appendingPathComponent(".swift-needle-tests")
    #endif
  }()
}
