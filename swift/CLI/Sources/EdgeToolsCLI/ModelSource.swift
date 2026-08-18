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
  public var quant: String?

  public init(
    location: Location,
    cacheDirectory: URL = ModelSource.defaultCacheDirectory,
    quant: String? = nil
  ) {
    self.location = location
    self.cacheDirectory = cacheDirectory
    self.quant = quant
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
    try await self.resolve(onDownloadStart: onDownloadStart) { repo, revision, cacheDirectory, globs in
      let hub = HubApi(downloadBase: cacheDirectory)
      guard let globs else {
        return try await hub.snapshot(from: Hub.Repo(id: repo, type: .models), revision: revision)
      }
      return try await hub.snapshot(
        from: Hub.Repo(id: repo, type: .models),
        revision: revision,
        matching: globs
      )
    }
  }

  public func resolve(
    onDownloadStart: @Sendable (String) -> Void = { _ in },
    snapshot: @Sendable (String, String, URL, [String]?) async throws -> URL
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
      onDownloadStart(repo)
      return try await snapshot(repo, revision, self.cacheDirectory, self.quantGlobs)
    }
  }

  /// The download globs narrowing a GGUF repo to one quantization, or nil to fetch everything.
  public var quantGlobs: [String]? {
    guard let quant = self.quant else { return nil }
    return ["*\(quant)*.gguf", "*.json", "*.jinja"]
  }
}
