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
}

// MARK: - Default Cache

extension ModelSource {
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
      if revision == "main", modelDirectoryIsPopulated(destination) {
        return destination
      }
      onDownloadStart(repo)
      return try await hub.snapshot(from: repository, revision: revision)
    }
  }
}

// MARK: - EdgeCLIError

public struct EdgeCLIError: Error, CustomStringConvertible {
  public let description: String

  public init(_ description: String) {
    self.description = description
  }
}

private func modelDirectoryIsPopulated(_ directory: URL) -> Bool {
  let contents = try? FileManager.default.contentsOfDirectory(atPath: directory.path())
  return !(contents ?? []).isEmpty
}
