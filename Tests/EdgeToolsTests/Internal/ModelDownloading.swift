import Foundation

#if canImport(Hub)
  import Hub

  func downloadNeedle() async throws -> URL {
    let hub = HubApi(downloadBase: URL.swiftEdgeToolsTestsDirectory)
    let repo = Hub.Repo(id: "Cactus-Compute/needle", type: .models)
    let destination = hub.localRepoLocation(repo)

    if FileManager.default.fileExists(atPath: destination.path) {
      print("=== Needle Model Already Downloaded At \(destination.path) ===")
      return destination
    }

    print("=== Downloading Needle Model From \(repo.id) ===")
    let url = try await hub.snapshot(from: repo)
    print("=== Finished Downloading Needle Model To \(url.path) ===")
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
