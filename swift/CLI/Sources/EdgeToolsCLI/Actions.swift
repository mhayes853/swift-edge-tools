import EdgeTools
import Foundation

// MARK: - Run

public func runModel(
  context: EdgeContext,
  source: ModelSource,
  requestedEngine: EngineKind? = nil,
  hardwareUnit: MLXHardwareUnit = .gpu,
  request: GenerationRequest,
  stream: StreamOption = .none,
  onOutput: @escaping @Sendable (String, String) -> Void = { _, _ in },
  onWarning: @escaping @Sendable (String) -> Void = { _ in }
) async throws -> RunReport {
  let loaded = try await LoadedModel.load(
    context: context,
    source: source,
    requestedEngine: requestedEngine,
    hardwareUnit: hardwareUnit,
    onWarning: onWarning
  )
  let request = try loaded.validate(request)
  let start = context.now()
  let printer = StreamPrinter(mode: stream, start: start, output: onOutput)
  let generation = try await loaded.generate(
    request,
    hardwareUnit: hardwareUnit,
    onToken: { printer.token($0) },
    onToolCall: { printer.toolCall($0) }
  )
  printer.finish()
  let peakMemory = context.peakMemory()

  return RunReport(
    model: loaded.detection.model.displayName,
    engine: loaded.runner.engine.rawValue,
    response: generation.response,
    wasStopped: generation.wasStopped,
    toolCalls: generation.toolCalls.map {
      RunReport.ToolCall(name: $0.name, arguments: $0.arguments)
    },
    metrics: RunReport.Metrics(
      load: loaded.loadDuration,
      endToEnd: start.duration(to: context.now()),
      prefill: generation.prefillMetrics,
      decode: generation.decodeMetrics,
      peakResident: peakMemory.resident,
      peakGPU: peakMemory.gpu
    )
  )
}

// MARK: - Benchmark

public func benchmarkModel(
  context: EdgeContext,
  source: ModelSource,
  requestedEngine: EngineKind? = nil,
  hardwareUnit: MLXHardwareUnit = .gpu,
  request: GenerationRequest,
  runs: Int,
  warmup: Int,
  onWarning: @escaping @Sendable (String) -> Void = { _ in },
  onProgress: @Sendable (Int, Int) -> Void = { _, _ in }
) async throws -> BenchReport {
  let loaded = try await LoadedModel.load(
    context: context,
    source: source,
    requestedEngine: requestedEngine,
    hardwareUnit: hardwareUnit,
    onWarning: onWarning
  )
  let request = try loaded.validate(request)

  for _ in 0..<warmup {
    _ = try await loaded.generate(request, hardwareUnit: hardwareUnit)
  }

  var samples = [BenchSample]()
  for index in 0..<runs {
    onProgress(index + 1, runs)
    await loaded.runner.reset()
    let start = context.now()
    let generation = try await loaded.generate(request, hardwareUnit: hardwareUnit)
    samples.append(
      BenchSample(
        endToEnd: start.duration(to: context.now()),
        prefill: generation.prefillMetrics,
        decode: generation.decodeMetrics,
        madeToolCalls: !generation.toolCalls.isEmpty
      )
    )
  }

  let peakMemory = context.peakMemory()
  return BenchReport(
    model: loaded.detection.model.displayName,
    engine: loaded.runner.engine.rawValue,
    runs: samples.count,
    warmup: warmup,
    samples: samples,
    peakResident: peakMemory.resident,
    peakGPU: peakMemory.gpu
  )
}

// MARK: - Info

public func inspectModel(
  context: EdgeContext,
  source: ModelSource,
  onWarning: @escaping @Sendable (String) -> Void = { _ in }
) async throws -> InfoReport {
  let directory = try await context.resolveDirectory(source) { repo in
    onWarning("downloading \(repo)...")
  }
  return InfoReport(detection: try context.detectModel(directory))
}

// MARK: - BenchSample

public struct BenchSample: Sendable {
  public let endToEnd: Duration
  public let prefill: EdgeToolsPrefillMetrics
  public let decode: EdgeToolsDecodeMetrics
  public let madeToolCalls: Bool
}
