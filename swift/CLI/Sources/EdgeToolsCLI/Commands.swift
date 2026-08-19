import ArgumentParser
import Foundation

// MARK: - EdgeCommand

public struct EdgeCommand: AsyncParsableCommand {
  public static let configuration = CommandConfiguration(
    commandName: "edge",
    abstract: "Run and benchmark on-device tool calling models.",
    subcommands: [RunCommand.self, BenchCommand.self, InfoCommand.self],
    defaultSubcommand: RunCommand.self
  )

  public init() {}
}

extension EdgeCommand {
  public static func main() async {
    initializeLlamaBackendIfRequested(arguments: CommandLine.arguments)
    await self.main(nil)
  }

  public static func run(
    arguments: [String],
    context: EdgeContext
  ) async throws {
    let parsed = try await self.asyncParseAsRoot(arguments)
    guard var command = parsed as? any EdgeSubcommand else {
      throw EdgeCLIError("Expected an edge subcommand.")
    }
    try await command.run(context: context)
  }
}

private protocol EdgeSubcommand: AsyncParsableCommand {
  mutating func run(context: EdgeContext) async throws
}

extension EdgeSubcommand {
  mutating func run() async throws {
    try await self.run(context: .live)
  }
}

// MARK: - RunCommand

struct RunCommand: EdgeSubcommand {
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

  func run(context: EdgeContext) async throws {
    let prompt = try self.generation.resolvedPrompt(input: context.readStandardInput)
    let tools = try self.generation.toolDefinitions()
    let stream = self.json ? StreamOption.none : self.stream
    let warning = warningHandler(json: self.json, context: context)
    let report = try await runModel(
      context: context,
      source: self.model.source,
      requestedEngine: self.model.engine,
      hardwareUnit: self.model.hardwareUnit,
      request: self.generation.request(prompt: prompt, tools: tools),
      stream: stream,
      onOutput: context.writeStandardOutput,
      onWarning: warning
    )
    context.writeStandardOutput(
      self.json
        ? try report.jsonText()
        : report.displayText(includingResponse: stream == .none),
      "\n"
    )
  }
}

// MARK: - BenchCommand

struct BenchCommand: EdgeSubcommand {
  static let configuration = CommandConfiguration(
    commandName: "bench",
    abstract: "Repeatedly generate and report the distribution of performance metrics."
  )

  @OptionGroup var model: ModelOptions
  @OptionGroup var generation: GenerationOptions

  @Option(name: .customLong("repeat-count"), help: "How many measured runs to perform.")
  var runs: Int = 10

  @Option(help: "How many unmeasured runs to perform first.")
  var warmup: Int = 2

  @Flag(help: "Emit a single JSON object instead of human-readable output.")
  var json = false

  func validate() throws {
    guard self.runs > 0 else {
      throw ValidationError("--repeat-count must be at least 1.")
    }
    guard self.warmup >= 0 else {
      throw ValidationError("--warmup cannot be negative.")
    }
  }

  func run(context: EdgeContext) async throws {
    let prompt = try self.generation.resolvedPrompt(input: context.readStandardInput)
    let tools = try self.generation.toolDefinitions()
    let showsProgress = !self.json && context.isStandardErrorTTY()
    let warning = warningHandler(json: self.json, context: context)
    let writeError = context.writeStandardError
    defer {
      if showsProgress {
        writeError("\u{1B}[2K", "")
      }
    }
    let report = try await benchmarkModel(
      context: context,
      source: self.model.source,
      requestedEngine: self.model.engine,
      hardwareUnit: self.model.hardwareUnit,
      request: self.generation.request(prompt: prompt, tools: tools),
      runs: self.runs,
      warmup: self.warmup,
      onWarning: warning,
      onProgress: { index, total in
        guard showsProgress else { return }
        writeError("run \(index)/\(total)", "\r")
      }
    )
    context.writeStandardOutput(self.json ? try report.jsonText() : report.displayText(), "\n")
  }
}

// MARK: - InfoCommand

struct InfoCommand: EdgeSubcommand {
  static let configuration = CommandConfiguration(
    commandName: "info",
    abstract: "Report what model and engines were detected, without running anything."
  )

  @OptionGroup var model: ModelSourceOptions

  @Flag(help: "Emit a single JSON object instead of human-readable output.")
  var json = false

  func run(context: EdgeContext) async throws {
    let warning = warningHandler(json: self.json, context: context)
    let report = try await inspectModel(
      context: context,
      source: self.model.source,
      onWarning: warning
    )
    context.writeStandardOutput(self.json ? try report.jsonText() : report.displayText(), "\n")
  }
}

private func warningHandler(
  json: Bool,
  context: EdgeContext
) -> @Sendable (String) -> Void {
  guard !json else { return { _ in } }
  let writeError = context.writeStandardError
  return { writeError("warning: \($0)", "\n") }
}
