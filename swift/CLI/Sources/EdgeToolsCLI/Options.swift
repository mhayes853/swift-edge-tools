import ArgumentParser
import EdgeTools
import Foundation

// MARK: - ModelOptions

struct ModelOptions: ParsableArguments {
  @Argument(help: "A Hugging Face repo, such as Cactus-Compute/needle.")
  var repo: String?

  @Option(help: "A local model directory, used instead of a Hugging Face repo.")
  var path: String?

  @Option(help: "The Hugging Face revision to download.")
  var revision: String = "main"

  @Option(name: .customLong("cache-dir"), help: "Where downloaded models are cached.")
  var cacheDirectory: String?

  @Option(help: "The engine to run with. Defaults to the best available for the model.")
  var engine: EngineKind?

  @Option(
    name: .customLong("hardware-unit"),
    help: "The hardware unit for MLX: cpu or gpu. Defaults to gpu."
  )
  var hardwareUnit: MLXHardwareUnit = .gpu

  init() {}

  func validate() throws {
    switch (self.repo, self.path) {
    case (nil, nil):
      throw ValidationError("Specify a Hugging Face repo or --path.")
    case (.some, .some):
      throw ValidationError("Specify either a Hugging Face repo or --path, not both.")
    default:
      break
    }
    if self.path != nil, self.revision != "main" || self.cacheDirectory != nil {
      throw ValidationError("--revision and --cache-dir do not apply to --path.")
    }
  }

  var source: ModelSource {
    if let path {
      return ModelSource(location: .filesystem(URL(fileURLWithPath: path)))
    }
    return ModelSource(
      location: .huggingFace(repo: self.repo ?? "", revision: self.revision),
      cacheDirectory: self.cacheDirectory.map { URL(fileURLWithPath: $0) }
        ?? ModelSource.defaultCacheDirectory
    )
  }
}

extension EngineKind: ExpressibleByArgument {}
extension MLXHardwareUnit: ExpressibleByArgument {}

// MARK: - GenerationOptions

struct GenerationOptions: ParsableArguments {
  @Option(name: .shortAndLong, help: "The user prompt. Read from stdin when omitted.")
  var prompt: String?

  @Option(name: .shortAndLong, help: "The system prompt.")
  var system: String = ""

  @Option(
    help: """
      A JSON file of tool definitions, in OpenAI function-calling format. Defaults to a small \
      built-in toolset.
      """
  )
  var tools: String?

  @Option(
    help: """
      The generation constraint: auto, unconstrained, a grammar file (.ebnf, .lark, .json), or an \
      inline <format>:<value>.
      """
  )
  var grammar: GrammarOption = .auto

  @Option(name: .customLong("max-tokens"), help: "The maximum number of tokens to decode.")
  var maxTokens: Int = 1024

  @Option(help: "The sampling temperature. 0 samples greedily.")
  var temperature: Float = 0

  @Option(name: .customLong("top-p"), help: "The nucleus sampling threshold.")
  var topP: Float = 1

  @Option(
    name: .customLong("tool-calls"),
    help: "Whether tool calls are required: auto, required, none."
  )
  var toolCalls: ToolCallsOption = .auto

  @Option(
    name: .customLong("max-tool-calls"),
    help: "The maximum number of tool calls to constrain to."
  )
  var maxToolCalls: Int?

  init() {}

  func validate() throws {
    if self.toolCalls == .none, self.maxToolCalls != nil {
      throw ValidationError("--max-tool-calls cannot be used with --tool-calls none.")
    }
    if let maximum = self.maxToolCalls, maximum < self.toolCalls.minimumCalls {
      throw ValidationError(
        "--max-tool-calls must be at least \(self.toolCalls.minimumCalls) for --tool-calls \(self.toolCalls.rawValue)."
      )
    }
  }

  func request(prompt: String, tools: [EdgeToolDefinition]) -> GenerationRequest {
    GenerationRequest(
      system: self.system,
      user: prompt,
      tools: tools,
      grammar: self.grammar,
      toolCallRange: self.toolCallRange,
      maxTokens: self.maxTokens,
      temperature: self.temperature,
      topP: self.topP
    )
  }

  private var toolCallRange: GrammarToolCallRange {
    let minimum = self.toolCalls.minimumCalls
    guard let maximum = self.maxToolCalls else {
      return self.toolCalls == .none ? .exact(0) : .unbounded(minimum: minimum)
    }
    return .bounded(minimum...maximum)
  }

  func resolvedPrompt(input: () -> String) throws -> String {
    if let prompt { return prompt }
    let trimmed = input().trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      throw ValidationError("No prompt. Pass --prompt or pipe one in on stdin.")
    }
    return trimmed
  }

  func toolDefinitions() throws -> [EdgeToolDefinition] {
    guard let tools else { return defaultToolDefinitions }
    return try ToolsFile(contentsOf: URL(fileURLWithPath: tools)).definitions
  }
}

// MARK: - ToolCallsOption

public enum ToolCallsOption: String, ExpressibleByArgument, CaseIterable, Sendable {
  case auto
  case required
  case none

  var minimumCalls: Int {
    switch self {
    case .auto, .none: 0
    case .required: 1
    }
  }
}

// MARK: - StreamOption

public enum StreamOption: String, ExpressibleByArgument, CaseIterable, Sendable {
  case tokens
  case events
  case none
}
