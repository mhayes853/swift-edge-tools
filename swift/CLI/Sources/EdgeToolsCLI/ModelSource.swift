import Foundation
import Hub

// MARK: - ModelSource

public struct ModelSource: Hashable, Sendable {
  public enum Location: Hashable, Sendable {
    case huggingFace(repo: String, revision: String)
    case filesystem(URL)
  }

  public var location: Location
  public var cacheDirectory: URL

  public init(location: Location, cacheDirectory: URL = ModelSource.defaultCacheDirectory) {
    self.location = location
    self.cacheDirectory = cacheDirectory
  }

  public static var defaultCacheDirectory: URL {
    let environment = ProcessInfo.processInfo.environment
    if let home = environment["HF_HOME"], !home.isEmpty {
      return URL(fileURLWithPath: home)
    }
    return FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface")
  }
}

// MARK: - Resolution

extension ModelSource {
  public func resolve(
    onDownloadStart: @Sendable (String) -> Void = { _ in }
  ) async throws -> URL {
    switch self.location {
    case .filesystem(let url):
      let standardized = url.standardizedFileURL
      var isDirectory: ObjCBool = false
      let exists = FileManager.default.fileExists(
        atPath: standardized.path(),
        isDirectory: &isDirectory
      )
      guard exists, isDirectory.boolValue else {
        throw EdgeCLIError("No model directory at \(standardized.path()).")
      }
      return standardized

    case .huggingFace(let repo, let revision):
      let hub = HubApi(downloadBase: self.cacheDirectory)
      let repository = Hub.Repo(id: repo, type: .models)
      let destination = hub.localRepoLocation(repository)
      if revision == "main", isPopulated(destination) {
        return destination
      }
      onDownloadStart(repo)
      return try await hub.snapshot(from: repository, revision: revision)
    }
  }
}

private func isPopulated(_ directory: URL) -> Bool {
  let contents = try? FileManager.default.contentsOfDirectory(atPath: directory.path())
  return !(contents ?? []).isEmpty
}
