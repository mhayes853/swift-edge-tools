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
    let stream = self.json ? StreamOption.none : self.stream
    let action = RunAction(
      context: .live,
      source: self.model.source,
      requestedEngine: self.model.engine,
      settings: self.generation.settings,
      stream: stream,
      quiet: self.json
    )
    let report = try await action(prompt: prompt, tools: tools)
    output(
      self.json
        ? try report.jsonText()
        : report.displayText(includingResponse: stream == .none)
    )
  }
}

// MARK: - BenchCommand

struct BenchCommand: AsyncParsableCommand {
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

  func run() async throws {
    let prompt = try self.generation.resolvedPrompt()
    let tools = try self.generation.toolDefinitions()
    let action = BenchAction(
      context: .live,
      source: self.model.source,
      requestedEngine: self.model.engine,
      settings: self.generation.settings,
      runs: self.runs,
      warmup: self.warmup,
      quiet: self.json
    )
    let showsProgress = !self.json && isatty(STDERR_FILENO) == 1
    let report = try await action(prompt: prompt, tools: tools) { index, total in
      guard showsProgress else { return }
      FileHandle.standardError.write(Data("run \(index)/\(total)\r".utf8))
    }
    if showsProgress {
      FileHandle.standardError.write(Data("\u{1B}[2K".utf8))
    }
    output(self.json ? try report.jsonText() : report.displayText())
  }
}

// MARK: - InfoCommand

struct InfoCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "info",
    abstract: "Report what model and engines were detected, without running anything."
  )

  @OptionGroup var model: ModelOptions

  @Flag(help: "Emit a single JSON object instead of human-readable output.")
  var json = false

  func run() async throws {
    let action = InfoAction(context: .live, source: self.model.source, quiet: self.json)
    let report = try await action()
    output(self.json ? try report.jsonText() : report.displayText())
  }
}
