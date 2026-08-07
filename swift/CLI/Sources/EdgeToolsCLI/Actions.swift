import EdgeTools
import Foundation

// MARK: - RunAction

public struct RunAction: Sendable {
  public var context: EdgeContext
  public var source: ModelSource
  public var requestedEngine: EngineKind?
  public var settings: GenerationSettings
  public var stream: StreamOption
  public var quiet: Bool

  public init(
    context: EdgeContext,
    source: ModelSource,
    requestedEngine: EngineKind? = nil,
    settings: GenerationSettings = GenerationSettings(),
    stream: StreamOption = .none,
    quiet: Bool = true
  ) {
    self.context = context
    self.source = source
    self.requestedEngine = requestedEngine
    self.settings = settings
    self.stream = stream
    self.quiet = quiet
  }

  public func callAsFunction(
    prompt: String,
    tools: [EdgeToolDefinition] = []
  ) async throws -> RunReport {
    let loaded = try await LoadedModel.load(
      context: self.context,
      source: self.source,
      requestedEngine: self.requestedEngine,
      quiet: self.quiet
    )
    let request = try loaded.makeRequest(
      settings: self.settings,
      prompt: prompt,
      tools: tools
    )

    let clock = ContinuousClock()
    let start = clock.now
    let printer = StreamPrinter(mode: self.stream, start: start)
    let generation = try await loaded.runner.generate(
      request,
      channel: EdgeToolsGenerationChannel(
        onToken: { printer.token($0) },
        onToolCall: { printer.toolCall($0) }
      )
    )
    printer.finish()
    let peakMemory = self.context.peakMemory()

    return RunReport(
      model: loaded.detection.model.displayName,
      engine: loaded.engine.rawValue,
      response: generation.response,
      wasStopped: generation.wasStopped,
      toolCalls: generation.toolCalls.map {
        RunReport.ToolCall(name: $0.name, arguments: $0.arguments)
      },
      metrics: RunReport.Metrics(
        load: loaded.loadDuration,
        endToEnd: start.duration(to: clock.now),
        prefill: generation.prefillMetrics,
        decode: generation.decodeMetrics,
        peakResident: peakMemory.resident,
        peakGPU: peakMemory.gpu
      )
    )
  }
}

// MARK: - BenchAction

public struct BenchAction: Sendable {
  public var context: EdgeContext
  public var source: ModelSource
  public var requestedEngine: EngineKind?
  public var settings: GenerationSettings
  public var runs: Int
  public var warmup: Int
  public var quiet: Bool

  public init(
    context: EdgeContext,
    source: ModelSource,
    requestedEngine: EngineKind? = nil,
    settings: GenerationSettings = GenerationSettings(),
    runs: Int = 10,
    warmup: Int = 2,
    quiet: Bool = true
  ) {
    self.context = context
    self.source = source
    self.requestedEngine = requestedEngine
    self.settings = settings
    self.runs = runs
    self.warmup = warmup
    self.quiet = quiet
  }

  public func callAsFunction(
    prompt: String,
    tools: [EdgeToolDefinition] = [],
    onProgress: @Sendable (Int, Int) -> Void = { _, _ in }
  ) async throws -> BenchReport {
    let loaded = try await LoadedModel.load(
      context: self.context,
      source: self.source,
      requestedEngine: self.requestedEngine,
      quiet: self.quiet
    )
    let request = try loaded.makeRequest(
      settings: self.settings,
      prompt: prompt,
      tools: tools
    )

    for _ in 0..<self.warmup {
      _ = try await loaded.runner.generate(request)
    }

    let clock = ContinuousClock()
    var samples = [BenchSample]()
    for index in 0..<self.runs {
      onProgress(index + 1, self.runs)
      await loaded.runner.reset()
      let start = clock.now
      let generation = try await loaded.runner.generate(request)
      samples.append(
        BenchSample(
          endToEnd: start.duration(to: clock.now),
          prefill: generation.prefillMetrics,
          decode: generation.decodeMetrics,
          madeToolCalls: !generation.toolCalls.isEmpty
        )
      )
    }

    let peakMemory = self.context.peakMemory()
    return BenchReport(
      model: loaded.detection.model.displayName,
      engine: loaded.engine.rawValue,
      runs: samples.count,
      warmup: self.warmup,
      samples: samples,
      peakResident: peakMemory.resident,
      peakGPU: peakMemory.gpu
    )
  }
}

// MARK: - InfoAction

public struct InfoAction: Sendable {
  public var context: EdgeContext
  public var source: ModelSource
  public var quiet: Bool

  public init(context: EdgeContext, source: ModelSource, quiet: Bool = true) {
    self.context = context
    self.source = source
    self.quiet = quiet
  }

  public func callAsFunction() async throws -> InfoReport {
    let directory = try await self.context.resolveDirectory(self.source) { repo in
      if !self.quiet { warn("downloading \(repo)...") }
    }
    return InfoReport(detection: try self.context.detectModel(directory))
  }
}

// MARK: - BenchSample

public struct BenchSample: Sendable {
  public let endToEnd: Duration
  public let prefill: EdgeToolsPrefillMetrics
  public let decode: EdgeToolsDecodeMetrics
  public let madeToolCalls: Bool
}
