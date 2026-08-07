import Foundation

// MARK: - EdgeContext

// The services that touch the network, the filesystem, or an inference engine. Commands take a
// context so tests can substitute all of them.
public struct EdgeContext: Sendable {
  public var resolveDirectory:
    @Sendable (ModelSource, @Sendable (String) -> Void) async throws -> URL
  public var detectModel: @Sendable (URL) throws -> ModelDetection
  public var makeRunner: @Sendable (ModelDetection, EngineKind) async throws -> EngineRunner
  public var peakMemory: @Sendable () -> PeakMemory

  public init(
    resolveDirectory:
      @escaping @Sendable (ModelSource, @Sendable (String) -> Void) async throws ->
      URL,
    detectModel: @escaping @Sendable (URL) throws -> ModelDetection,
    makeRunner: @escaping @Sendable (ModelDetection, EngineKind) async throws -> EngineRunner,
    peakMemory: @escaping @Sendable () -> PeakMemory
  ) {
    self.resolveDirectory = resolveDirectory
    self.detectModel = detectModel
    self.makeRunner = makeRunner
    self.peakMemory = peakMemory
  }
}

// MARK: - Live

extension EdgeContext {
  public static let live = Self(
    resolveDirectory: { source, onDownloadStart in
      try await source.resolve(onDownloadStart: onDownloadStart)
    },
    detectModel: { try ModelDetection.detect(in: $0) },
    makeRunner: { try await EngineRunner(detection: $0, engine: $1) },
    peakMemory: { .current }
  )
}
