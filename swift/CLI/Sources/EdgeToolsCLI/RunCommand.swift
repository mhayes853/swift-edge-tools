import ArgumentParser
import EdgeTools
import Foundation

// MARK: - RunCommand

struct RunCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "run",
    abstract: "Generate a response, and any tool calls, from a model."
  )

  @OptionGroup var model: ModelOptions
  @OptionGroup var generation: GenerationOptions

  @Option(help: "How output is streamed: tokens, events, none.")
  var stream: StreamOption = .tokens

  @Flag(help: "Emit a single JSON object instead of human-readable output.")
  var json = false

  func run() async throws {
    let prompt = try self.generation.resolvedPrompt()
    let tools = try self.generation.toolDefinitions()
    let loaded = try await loadModel(model: self.model, quiet: self.json)

    let stream = self.json ? StreamOption.none : self.stream
    let request = try makeRequest(
      options: self.generation,
      prompt: prompt,
      tools: tools,
      loaded: loaded,
      quiet: self.json
    )

    let clock = ContinuousClock()
    let start = clock.now
    let printer = StreamPrinter(mode: stream, start: start)
    let generation = try await loaded.runner.generate(
      request,
      channel: EdgeToolsGenerationChannel(
        onToken: { printer.token($0) },
        onToolCall: { printer.toolCall($0) }
      )
    )
    let metrics = RunMetrics(
      loadDuration: loaded.loadDuration,
      generationDuration: start.duration(to: clock.now),
      prefill: generation.prefillMetrics,
      decode: generation.decodeMetrics,
      peakResidentBytes: peakResidentBytes(),
      peakGPUBytes: peakGPUBytes()
    )
    printer.finish()

    if self.json {
      output(try runJSON(generation: generation, loaded: loaded, metrics: metrics))
    } else {
      printRunReport(generation: generation, loaded: loaded, metrics: metrics, stream: stream)
    }
  }
}

// MARK: - LoadedModel

struct LoadedModel {
  var detection: ModelDetection
  var engine: EngineKind
  var runner: any EdgeRunner
  var loadDuration: Duration
}

func loadModel(model options: ModelOptions, quiet: Bool) async throws -> LoadedModel {
  let clock = ContinuousClock()
  let start = clock.now
  let directory = try await options.source.resolve(
    onDownloadStart: { repo in
      if !quiet { FileHandle.standardError.write(Data("Downloading \(repo)...\n".utf8)) }
    }
  )
  let detection = try ModelDetection.detect(in: directory)
  guard let engine = options.engine ?? detection.defaultEngine else {
    let experimental = detection.engines.filter(\.isExperimental).map(\.rawValue)
    throw EdgeCLIError(
      """
      No usable engine for \(detection.model.displayName) in \(directory.path()). \
      \(experimental.isEmpty
        ? "Supported engines: \(detection.model.supportedEngines.map(\.rawValue).joined(separator: ", "))."
        : "Select one explicitly with --engine: \(experimental.joined(separator: ", ")).")
      """
    )
  }
  guard detection.engines.contains(engine) else {
    throw EdgeCLIError(
      """
      The \(engine.rawValue) engine has no weights for \(detection.model.displayName) here. \
      Available: \(detection.engines.map(\.rawValue).joined(separator: ", ")).
      """
    )
  }
  if engine.isExperimental, !quiet {
    warn("the \(engine.rawValue) engine is experimental.")
  }
  let runner = try await makeRunner(detection: detection, engine: engine)
  return LoadedModel(
    detection: detection,
    engine: engine,
    runner: runner,
    loadDuration: start.duration(to: clock.now)
  )
}

func makeRequest(
  options: GenerationOptions,
  prompt: String,
  tools: [EdgeToolDefinition],
  loaded: LoadedModel,
  quiet: Bool
) throws -> GenerationRequest {
  guard options.temperature == 0 || loaded.runner.supportsSampling else {
    throw EdgeCLIError(
      """
      \(loaded.detection.model.displayName) on \(loaded.engine.rawValue) always samples greedily; \
      --temperature and --top-p do not apply.
      """
    )
  }
  return GenerationRequest(
    system: options.system,
    user: prompt,
    tools: tools,
    grammar: try resolveGrammar(options.grammar, loaded: loaded, quiet: quiet),
    toolCallRange: options.toolCallRange,
    maxTokens: options.maxTokens,
    temperature: options.temperature,
    topP: options.topP
  )
}

/// Generic models have no tool call grammar to build, so `auto` degrades to unconstrained rather
/// than failing.
private func resolveGrammar(
  _ grammar: GrammarOption,
  loaded: LoadedModel,
  quiet: Bool
) throws -> GrammarOption {
  if grammar == .auto, loaded.detection.model.isGenericFallback {
    if !quiet {
      warn("no tool call grammar for this architecture — decoding unconstrained.")
    }
    return .unconstrained
  }
  guard grammar == .auto || loaded.runner.supportsCustomGrammar else {
    throw EdgeCLIError(
      """
      \(loaded.detection.model.displayName) on \(loaded.engine.rawValue) only supports \
      `--grammar auto`; its generate parameters expose a tool call range rather than a full \
      generation constraint.
      """
    )
  }
  return grammar
}

func warn(_ message: String) {
  FileHandle.standardError.write(Data("warning: \(message)\n".utf8))
}
