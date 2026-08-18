import Foundation

// MARK: - EdgeContext

// The services that touch the process, network, filesystem, or an inference engine. Commands take
// a context so tests can substitute all of them.
public struct EdgeContext: Sendable {
  public var resolveDirectory:
    @Sendable (ModelSource, @Sendable (String) -> Void) async throws -> URL
  public var detectModel: @Sendable (URL, String?) throws -> ModelDetection
  public var makeRunner:
    @Sendable (ModelDetection, EngineKind, MLXHardwareUnit) async throws -> EngineRunner
  public var peakMemory: @Sendable () -> PeakMemory
  public var now: @Sendable () -> ContinuousClock.Instant
  public var readStandardInput: @Sendable () -> String
  public var writeStandardOutput: @Sendable (String, String) -> Void
  public var writeStandardError: @Sendable (String, String) -> Void
  public var isStandardErrorTTY: @Sendable () -> Bool

  public init(
    resolveDirectory:
      @escaping @Sendable (ModelSource, @Sendable (String) -> Void) async throws ->
      URL,
    detectModel: @escaping @Sendable (URL, String?) throws -> ModelDetection,
    makeRunner:
      @escaping @Sendable (ModelDetection, EngineKind, MLXHardwareUnit) async throws ->
      EngineRunner,
    peakMemory: @escaping @Sendable () -> PeakMemory,
    now: @escaping @Sendable () -> ContinuousClock.Instant = { ContinuousClock().now },
    readStandardInput: @escaping @Sendable () -> String,
    writeStandardOutput: @escaping @Sendable (String, String) -> Void,
    writeStandardError: @escaping @Sendable (String, String) -> Void,
    isStandardErrorTTY: @escaping @Sendable () -> Bool
  ) {
    self.resolveDirectory = resolveDirectory
    self.detectModel = detectModel
    self.makeRunner = makeRunner
    self.peakMemory = peakMemory
    self.now = now
    self.readStandardInput = readStandardInput
    self.writeStandardOutput = writeStandardOutput
    self.writeStandardError = writeStandardError
    self.isStandardErrorTTY = isStandardErrorTTY
  }
}

// MARK: - Live

extension EdgeContext {
  public static let live = Self(
    resolveDirectory: { source, onDownloadStart in
      try await source.resolve(onDownloadStart: onDownloadStart)
    },
    detectModel: { try ModelDetection.detect(in: $0, quant: $1) },
    makeRunner: {
      try await EngineRunner(detection: $0, engine: $1, hardwareUnit: $2)
    },
    peakMemory: { .current },
    readStandardInput: { AnyIterator { readLine(strippingNewline: false) }.joined() },
    writeStandardOutput: { output($0, terminator: $1) },
    writeStandardError: {
      guard let data = ($0 + $1).data(using: .utf8) else { return }
      try? FileHandle.standardError.write(contentsOf: data)
    },
    isStandardErrorTTY: { isatty(STDERR_FILENO) == 1 }
  )
}
